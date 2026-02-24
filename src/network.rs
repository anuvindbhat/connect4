//! Networking module for Connect4.
//!
//! This module provides a robust, asynchronous networking layer for peer-to-peer
//! Connect4 games over a local area network (LAN). It handles service discovery,
//! connection management, and a custom synchronization protocol.
//!
//! ### Discovery Mechanism
//! Peer discovery is implemented using UDP broadcasts. Hosts periodically
//! broadcast [`DiscoveryInfo`] packets on a fixed port ([`UDP_PORT`]). Clients
//! listen for these packets to populate a list of available games.
//!
//! ### Synchronization Protocol
//! Once a connection is established over TCP, the peers engage in a symmetric
//! handshake to agree on game parameters (board size, player colors).
//! Communication is framed using a length-delimited codec to ensure reliable
//! message boundaries.
//!
//! ### Frame Format
//! Every TCP message is prefixed with a 4-byte big-endian length header,
//! followed by a [Postcard](https://github.com/jamesmunns/postcard) serialized
//! [`RemoteMessage`].
//!
//! ### Heartbeats & Reliability
//! To maintain connection health and measure latency (RTT), the module
//! implements a heartbeat mechanism using `Ping` and `Pong` messages.

use crate::config::{MAX_CHAT_LEN, MAX_FRAME_SIZE, MAX_NAME_LEN, NETWORK_TIMEOUT_MS};
use crate::types::Player;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use socket2::{Domain, Protocol, Socket, Type};
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use tracing::{debug, error, info, warn};

/// Truncates a string to a maximum length in characters.
///
/// This is used to sanitize user-provided strings (like names or chat messages)
/// before they are displayed or processed further.
fn truncate_string(s: &mut String, max_chars: usize) {
    if s.chars().count() > max_chars {
        let truncated: String = s.chars().take(max_chars).collect();
        *s = truncated;
    }
}

/// Errors that can occur during networking operations.
#[derive(thiserror::Error, Debug)]
pub enum NetworkError {
    /// An underlying I/O error occurred (e.g., connection reset, timed out).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Failed to serialize or deserialize a network message.
    #[error("Serialization error: {0}")]
    Serialization(#[from] postcard::Error),
    /// A violation of the expected network protocol occurred.
    #[error("Protocol error: {0}")]
    Protocol(String),
    /// A general networking error with a descriptive message.
    #[error("Network error: {0}")]
    Other(String),
}

/// A specialized Result type for networking operations.
pub type Result<T> = std::result::Result<T, NetworkError>;

// ========================================================================================
// NETWORKING PROTOCOL
// ========================================================================================

/// Current version of the network protocol.
///
/// Peers with mismatched versions will refuse to connect to ensure compatibility.
pub const PROTOCOL_VERSION: u32 = 1;

/// Fixed port used for UDP discovery broadcasts.
pub const UDP_PORT: u16 = 4445;

/// Identifies all available IPv4 broadcast addresses on the system.
///
/// This includes the limited broadcast (255.255.255.255) and directed
/// subnet broadcasts (e.g., 192.168.1.255) to ensure cross-platform
/// discovery visibility regardless of OS routing metrics.
fn get_broadcast_targets() -> Vec<SocketAddr> {
    use network_interface::{Addr, NetworkInterface, NetworkInterfaceConfig};

    let mut targets = vec![SocketAddr::new(
        std::net::IpAddr::V4(std::net::Ipv4Addr::BROADCAST),
        UDP_PORT,
    )];

    if let Ok(interfaces) = NetworkInterface::show() {
        for iface in interfaces {
            for addr in iface.addr {
                if let Addr::V4(v4_addr) = addr
                    && let Some(broadcast) = v4_addr.broadcast
                {
                    let addr = SocketAddr::new(std::net::IpAddr::V4(broadcast), UDP_PORT);
                    if !targets.contains(&addr) {
                        targets.push(addr);
                    }
                }
            }
        }
    }
    targets
}

/// Information broadcasted over the local network to allow peers to find hosts.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiscoveryInfo {
    /// Unique identifier for this game session to distinguish it from others.
    pub instance_id: u64,
    /// The display name of the hosting player.
    pub local_name: String,
    /// The protocol version of the host.
    pub protocol_version: u32,
    /// The address of the host's TCP listener.
    pub addr: SocketAddr,
    /// The dimensions of the game board (width, height).
    pub board_size: (u32, u32),
    /// The color assigned to the host.
    pub host_player: Player,
}

/// Messages exchanged between peers during a remote game.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RemoteMessage {
    /// Sent at the start of a connection to negotiate game parameters.
    Handshake {
        /// Protocol version being used.
        version: u32,
        /// Name of the player sending the handshake.
        player_name: String,
        /// Negotiated board dimensions.
        board_size: (u32, u32),
        /// Player color assigned to the sender.
        local_player: Player,
    },
    /// Sent when a player makes a move.
    Move {
        /// The column index (0-based) where the piece was dropped.
        col: u32,
        /// The hash of the board state after the move for validation.
        state_hash: u128,
    },
    /// A chat message sent to the peer.
    Chat(String),
    /// A heartbeat request to measure latency and keep the connection alive.
    Ping,
    /// A response to a Ping message.
    Pong,
    /// Notifies the peer that the sender is leaving the game.
    Quit,
}

/// The role of a node in a point-to-point connection.
#[derive(Debug, Clone)]
pub enum ConnectionRole {
    /// The node is hosting the game and providing authoritative settings.
    Host {
        /// The host player's name.
        player_name: String,
        /// The dimensions of the board being used.
        board_size: (u32, u32),
        /// The color assigned to the host.
        local_player: Player,
    },
    /// The node is connecting to a host and accepting its settings.
    Client {
        /// The guest player's name.
        player_name: String,
    },
}

/// The outcome of a successful connection handshake.
#[derive(Debug)]
pub struct HandshakeResult {
    /// The name of the remote peer.
    pub peer_name: String,
    /// The negotiated board size.
    pub board_size: (u32, u32),
    /// The color assigned to the remote peer.
    pub peer_player: Player,
}

/// Performs a symmetric protocol handshake for either Host or Client roles.
///
/// This function ensures both peers agree on game configuration before starting.
/// It returns a [`HandshakeResult`] on success.
///
/// # Errors
/// Returns [`NetworkError::Protocol`] if the handshake fails (e.g., version mismatch,
/// color conflict).
async fn perform_handshake(
    framed: &mut Framed<TcpStream, LengthDelimitedCodec>,
    role: ConnectionRole,
) -> Result<HandshakeResult> {
    tracing::debug!("Performing handshake as role: {:?}", role);
    match role {
        ConnectionRole::Host {
            player_name,
            board_size,
            local_player,
        } => {
            // 1. Host sends authoritative configuration
            send_handshake(framed, player_name, board_size, local_player).await?;

            // 2. Wait for guest to echo configuration
            let (peer_name, echoed_size, peer_player) = recv_handshake(framed).await?;

            if peer_player == local_player {
                tracing::warn!(
                    "Handshake failed: Color conflict detected (both players Red/Yellow)"
                );
                return Err(NetworkError::Protocol("Color conflict detected".into()));
            }
            if echoed_size != board_size {
                tracing::warn!(
                    "Handshake failed: Board size mismatch. Authoritative {:?}, Echoed {:?}",
                    board_size,
                    echoed_size
                );
                return Err(NetworkError::Protocol(format!(
                    "Board size mismatch: authoritative {board_size:?}, echoed {echoed_size:?}",
                )));
            }

            tracing::debug!("Handshake successful (Host)");
            Ok(HandshakeResult {
                peer_name,
                board_size,
                peer_player,
            })
        }
        ConnectionRole::Client { player_name } => {
            // 1. Client waits for host configuration
            let (peer_name, host_size, host_player) = recv_handshake(framed).await?;

            // 2. Send client response, echoing host's settings
            send_handshake(framed, player_name, host_size, host_player.other()).await?;

            tracing::debug!("Handshake successful (Client)");
            Ok(HandshakeResult {
                peer_name,
                board_size: host_size,
                peer_player: host_player,
            })
        }
    }
}

/// Sends a handshake message containing local game configuration.
///
/// # Errors
/// Returns [`NetworkError::Io`] if the transmission fails.
async fn send_handshake(
    framed: &mut Framed<TcpStream, LengthDelimitedCodec>,
    player_name: String,
    board_size: (u32, u32),
    local_player: Player,
) -> Result<()> {
    tracing::debug!("Sending handshake message to peer");
    let msg = RemoteMessage::Handshake {
        version: PROTOCOL_VERSION,
        player_name,
        board_size,
        local_player,
    };
    let data = postcard::to_stdvec(&msg)?;
    framed
        .send(tokio_util::bytes::Bytes::from(data))
        .await
        .map_err(NetworkError::Io)
}

/// Waits for and validates a handshake message from a remote peer.
///
/// This function blocks until a [`RemoteMessage::Handshake`] is received,
/// up to a configured timeout ([`NETWORK_TIMEOUT_MS`]).
///
/// # Errors
/// Returns [`NetworkError::Protocol`] if:
/// - The handshake times out.
/// - The peer closes the connection prematurely.
/// - The protocol version mismatches.
/// - An unexpected message type is received.
async fn recv_handshake(
    framed: &mut Framed<TcpStream, LengthDelimitedCodec>,
) -> Result<(String, (u32, u32), Player)> {
    tracing::debug!("Waiting for handshake message from peer...");
    let bytes = tokio::time::timeout(Duration::from_millis(NETWORK_TIMEOUT_MS), framed.next())
        .await
        .map_err(|_| {
            tracing::warn!("Handshake timeout after {}ms", NETWORK_TIMEOUT_MS);
            NetworkError::Protocol("Handshake timed out".to_string())
        })?
        .ok_or_else(|| {
            tracing::warn!("Handshake stream closed prematurely");
            NetworkError::Protocol("Handshake closed by peer".to_string())
        })?
        .map_err(NetworkError::Io)?;

    if let RemoteMessage::Handshake {
        version,
        player_name,
        board_size,
        local_player,
    } = postcard::from_bytes::<RemoteMessage>(&bytes)?
    {
        if version != PROTOCOL_VERSION {
            tracing::warn!(
                "Handshake version mismatch: expected {}, got {}",
                PROTOCOL_VERSION,
                version
            );
            return Err(NetworkError::Protocol(format!(
                "Version mismatch: expected {PROTOCOL_VERSION}, got {version}",
            )));
        }
        tracing::debug!(
            "Handshake received: name={}, size={:?}, player={:?}",
            player_name,
            board_size,
            local_player
        );
        let mut name = player_name;
        truncate_string(&mut name, MAX_NAME_LEN);
        Ok((name, board_size, local_player))
    } else {
        tracing::warn!("Unexpected message type during handshake");
        Err(NetworkError::Protocol(
            "Unexpected message type during handshake".to_string(),
        ))
    }
}

// ========================================================================================
// NETWORK MANAGER
// ========================================================================================

/// Commands that can be sent to the background network task.
#[derive(Debug)]
pub enum NetworkCommand {
    /// Starts hosting a game and broadcasting presence on the local network.
    StartHosting {
        /// Unique ID for this host instance.
        instance_id: u64,
        /// The name of the hosting player.
        player_name: String,
        /// The board size to use for the game.
        board_size: (u32, u32),
        /// The color the host will use.
        host_player: Player,
    },
    /// Stops any active hosting or discovery and closes current connections.
    StopDiscovery,
    /// Attempts to connect to a remote host.
    Connect(SocketAddr, String),
    /// Sends a message to the currently connected peer.
    Send(RemoteMessage),
}

/// Events emitted by the network layer to the application.
#[derive(Debug)]
pub enum NetworkEvent {
    /// A host has been discovered on the local network.
    HostDiscovered(DiscoveryInfo),
    /// A peer has initiated a connection (TCP level).
    PeerConnected(String),
    /// Handshake is complete and the session is ready for gameplay.
    Ready {
        /// Name of the remote peer.
        peer_name: String,
        /// Board size agreed upon during handshake.
        board_size: (u32, u32),
        /// Player color assigned to the remote peer.
        peer_player: Player,
    },
    /// A move has been received from the peer.
    MoveReceived {
        /// The column index of the move.
        col: u32,
        /// The hash of the board state after the move.
        state_hash: u128,
    },
    /// A chat message has been received.
    ChatReceived(String),
    /// Updated round-trip time (latency) in milliseconds.
    Latency(u64),
    /// The connection was lost or the peer quit.
    Disconnected(String),
    /// An error occurred in the networking layer.
    Error(String),
}

/// A high-level manager for Connect4 networking.
///
/// `NetworkManager` provides a message-passing interface to a background task
/// that handles all I/O, discovery, and protocol logic. It uses MPSC channels
/// for bidirectional communication with the rest of the application.
pub struct NetworkManager {
    /// Channel for sending commands to the background task.
    cmd_tx: mpsc::Sender<NetworkCommand>,
    /// Channel for receiving events from the background task.
    event_rx: mpsc::Receiver<NetworkEvent>,
}

impl NetworkManager {
    /// Starts the background networking task and returns a new `NetworkManager`.
    #[must_use]
    pub fn start() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(100);
        let (event_tx, event_rx) = mpsc::channel(100);
        tokio::spawn(async move {
            info!("Network loop starting");
            if let Err(e) = run_network_loop(cmd_rx, event_tx).await {
                error!("Network loop error: {e}");
            }
            info!("Network loop task exited");
        });
        Self { cmd_tx, event_rx }
    }

    /// Sends a command to the networking task asynchronously.
    ///
    /// # Errors
    /// Returns an error if the background network task has panicked or the channel is closed.
    pub async fn send(&self, cmd: NetworkCommand) -> Result<()> {
        self.cmd_tx
            .send(cmd)
            .await
            .map_err(|e| NetworkError::Other(format!("Network task channel closed: {e}")))
    }

    /// Attempts to send a command without blocking.
    /// Useful for calling from synchronous UI code.
    pub fn try_send(&self, cmd: NetworkCommand) {
        if let Err(e) = self.cmd_tx.try_send(cmd) {
            warn!("Failed to dispatch network command {:?}: {}", e, e);
        }
    }

    /// Attempts to receive a networking event without blocking.
    ///
    /// Returns `None` if no events are currently available in the queue.
    pub fn try_recv(&mut self) -> Option<NetworkEvent> {
        self.event_rx.try_recv().ok()
    }
}

// ========================================================================================
// NETWORK LOOP STATE MACHINE
// ========================================================================================

/// Encapsulates the mutable state of the background network loop.
///
/// This state machine manages active connections, discovery, and heartbeat
/// status. It separates the high-level logic from the async event loop.
struct NetworkLoopState {
    /// Listens for incoming TCP connections when hosting.
    tcp_listener: Option<TcpListener>,
    /// Stores the current game discovery information and its pre-serialized
    /// bytes to optimize broadcast operations.
    hosting_info: Option<(DiscoveryInfo, Vec<u8>)>,
    /// The active length-delimited TCP stream for communication with a peer.
    stream: Option<Framed<TcpStream, LengthDelimitedCodec>>,
    /// Timestamp of the last sent Ping message to calculate RTT.
    last_ping_sent: Option<std::time::Instant>,
    /// The color assigned to the local host (Authoritative during handshake).
    current_host_player: Player,
    /// Channel used to emit events to the [`NetworkManager`].
    event_tx: mpsc::Sender<NetworkEvent>,
}

impl NetworkLoopState {
    /// Creates a new, empty network state.
    fn new(event_tx: mpsc::Sender<NetworkEvent>) -> Self {
        Self {
            tcp_listener: None,
            hosting_info: None,
            stream: None,
            last_ping_sent: None,
            current_host_player: Player::Red,
            event_tx,
        }
    }

    /// Resets the session state, closing any active connections and stopping hosting.
    fn cleanup_session(&mut self) {
        self.stream = None;
        self.tcp_listener = None;
        self.hosting_info = None;
        self.last_ping_sent = None;
        self.current_host_player = Player::Red;
    }

    /// Processes an incoming UDP discovery packet from the network.
    ///
    /// If the packet is valid and matches the protocol version, it triggers a
    /// [`NetworkEvent::HostDiscovered`] event.
    async fn handle_udp_discovery(&self, res: std::io::Result<(usize, SocketAddr)>, buf: &[u8]) {
        if let Ok((len, mut src_addr)) = res
            && let Ok(mut info) = postcard::from_bytes::<DiscoveryInfo>(&buf[..len])
        {
            if info.protocol_version != PROTOCOL_VERSION {
                warn!(
                    "Received discovery packet with incompatible protocol version: expected {}, got {}",
                    PROTOCOL_VERSION, info.protocol_version
                );
                return;
            }

            truncate_string(&mut info.local_name, MAX_NAME_LEN);
            debug!(
                "Received broadcast from {}: {:?}",
                info.local_name, info.addr
            );
            // We use the source IP from the UDP packet but the port from the payload
            // because the UDP broadcast port is different from the TCP game port.
            src_addr.set_port(info.addr.port());
            info.addr = src_addr;
            if let Err(e) = self.event_tx.send(NetworkEvent::HostDiscovered(info)).await {
                error!("Failed to relay host discovery: {e}");
            }
        }
    }

    /// Broadcasts presence information to a specific target address.
    async fn handle_udp_broadcast(&self, socket: &UdpSocket, broadcast_addr: SocketAddr) {
        if let Some((_, ref data)) = self.hosting_info {
            tracing::debug!("Broadcasting presence to {}", broadcast_addr);
            if let Err(e) = socket.send_to(data, broadcast_addr).await {
                warn!("Global broadcast failed: {e}");
            }
        }
    }

    /// Handles an incoming TCP connection request.
    ///
    /// This method performs the initial socket setup and initiates the handshake.
    async fn handle_tcp_accept(&mut self, res: Option<(TcpStream, SocketAddr)>) {
        let Some((tcp_stream, addr)) = res else {
            return;
        };

        tracing::info!("Accepted incoming TCP connection from guest at {}", addr);
        if let Err(e) = tcp_stream.set_nodelay(true) {
            error!("Failed to set TCP nodelay: {e}");
            return;
        }

        let mut framed = Framed::new(
            tcp_stream,
            LengthDelimitedCodec::builder()
                .max_frame_length(MAX_FRAME_SIZE)
                .new_codec(),
        );
        if let Some((ref info, _)) = self.hosting_info {
            let role = ConnectionRole::Host {
                player_name: info.local_name.clone(),
                board_size: info.board_size,
                local_player: self.current_host_player,
            };

            tracing::debug!("Initiating guest handshake for connection from {}", addr);
            match perform_handshake(&mut framed, role).await {
                Ok(result) => {
                    info!(
                        "Guest handshake successful from '{}', session ready",
                        result.peer_name
                    );
                    self.stream = Some(framed);
                    self.tcp_listener = None;
                    self.hosting_info = None;
                    if let Err(e) = self
                        .event_tx
                        .send(NetworkEvent::PeerConnected(result.peer_name))
                        .await
                    {
                        error!("Failed to relay peer connection: {e}");
                    }
                }
                Err(e) => {
                    warn!("Handshake failed for guest connection from {}: {}", addr, e);
                }
            }
        }
    }

    /// Processes an incoming [`NetworkCommand`].
    ///
    /// Returns `false` if the command signals that the loop should terminate.
    async fn handle_command(&mut self, cmd: Option<NetworkCommand>) -> bool {
        let Some(cmd) = cmd else {
            info!("Network command channel closed, signaling exit");
            return false;
        };

        debug!("Network command: {:?}", cmd);
        match cmd {
            NetworkCommand::StartHosting {
                instance_id,
                player_name,
                board_size,
                host_player,
            } => {
                self.start_hosting(instance_id, player_name, board_size, host_player)
                    .await;
            }
            NetworkCommand::Connect(addr, player_name) => {
                self.connect_as_client(addr, player_name).await;
            }
            NetworkCommand::Send(msg) => {
                self.send_message(msg).await;
            }
            NetworkCommand::StopDiscovery => {
                info!("Stopping discovery/hosting/session");
                self.cleanup_session();
            }
        }
        true
    }

    /// Binds a TCP listener and initializes discovery info to start hosting.
    async fn start_hosting(
        &mut self,
        instance_id: u64,
        player_name: String,
        board_size: (u32, u32),
        host_player: Player,
    ) {
        info!(
            "Starting host: name={}, size={}x{}, player={:?}",
            player_name, board_size.0, board_size.1, host_player
        );
        self.cleanup_session();
        self.current_host_player = host_player;

        match TcpListener::bind("0.0.0.0:0").await {
            Ok(listener) => {
                let actual_port = listener.local_addr().map_or(0, |a| a.port());
                tracing::info!("TCP listener bound to port {}", actual_port);
                let info = DiscoveryInfo {
                    instance_id,
                    local_name: player_name,
                    protocol_version: PROTOCOL_VERSION,
                    addr: format!("0.0.0.0:{actual_port}")
                        .parse()
                        .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], 0))),
                    board_size,
                    host_player,
                };
                if let Ok(data) = postcard::to_stdvec(&info) {
                    self.hosting_info = Some((info, data));
                    self.tcp_listener = Some(listener);
                }
            }
            Err(e) => {
                error!("Failed to bind TCP listener: {e}");
                self.send_error(format!("Failed to start host: {e}")).await;
            }
        }
    }

    /// Initiates a connection to a remote host as a client.
    async fn connect_as_client(&mut self, addr: SocketAddr, player_name: String) {
        info!("Connecting to host at {} as {}", addr, player_name);
        self.cleanup_session();

        match connect_to_host(addr, player_name, self.event_tx.clone()).await {
            Ok(s) => {
                info!("Successfully connected and handshaked with {}", addr);
                self.stream = Some(s);
            }
            Err(e) => {
                error!("Failed to connect to {}: {}", addr, e);
                self.send_error(format!("Connect failed: {e}")).await;
            }
        }
    }

    /// Sends a serialized [`RemoteMessage`] over the active TCP stream.
    async fn send_message(&mut self, msg: RemoteMessage) {
        if let Some(ref mut s) = self.stream {
            tracing::debug!("Sending remote message: {:?}", msg);
            if let Ok(data) = postcard::to_stdvec(&msg)
                && let Err(e) = s.send(tokio_util::bytes::Bytes::from(data)).await
            {
                error!("Failed to send custom message: {e}");
            }
        }
    }

    /// Sends a Ping message to the peer to check connectivity and measure latency.
    async fn handle_heartbeat(&mut self) {
        if let Some(ref mut s) = self.stream {
            tracing::trace!("Sending heartbeat ping");
            if let Ok(data) = postcard::to_stdvec(&RemoteMessage::Ping) {
                self.last_ping_sent = Some(std::time::Instant::now());
                if let Err(e) = s.send(tokio_util::bytes::Bytes::from(data)).await {
                    warn!("Ping send failed: {e}");
                    self.cleanup_session();
                    if let Err(send_err) = self
                        .event_tx
                        .send(NetworkEvent::Disconnected("Ping failed".to_string()))
                        .await
                    {
                        error!("Failed to relay ping failure disconnect: {send_err}");
                    }
                }
            }
        }
    }

    /// Processes raw bytes received from the peer's TCP stream.
    async fn handle_tcp_input(
        &mut self,
        res: Option<std::io::Result<tokio_util::bytes::BytesMut>>,
    ) {
        match res {
            Some(Ok(bytes)) => {
                if let Ok(msg) = postcard::from_bytes::<RemoteMessage>(&bytes) {
                    self.process_incoming_message(msg).await;
                }
            }
            Some(Err(e)) => {
                error!("TCP receive error: {e}");
                // Don't close immediately on transient errors?
                // Actually, framing errors usually mean stream is toast.
            }
            None => {
                info!("TCP stream closed or timed out");
                self.cleanup_session();
                if let Err(e) = self
                    .event_tx
                    .send(NetworkEvent::Disconnected(
                        "Connection lost or timed out".to_string(),
                    ))
                    .await
                {
                    error!("Failed to relay connection loss disconnect: {e}");
                }
            }
        }
    }

    /// Decodes and handles a high-level [`RemoteMessage`].
    async fn process_incoming_message(&mut self, msg: RemoteMessage) {
        tracing::debug!("Received remote message: {:?}", msg);
        match msg {
            RemoteMessage::Ping => {
                if let Some(ref mut s) = self.stream {
                    tracing::trace!("Responding to ping with pong");
                    if let Ok(data) = postcard::to_stdvec(&RemoteMessage::Pong)
                        && let Err(e) = s.send(tokio_util::bytes::Bytes::from(data)).await
                    {
                        warn!("Failed to respond to ping: {e}");
                    }
                }
            }
            RemoteMessage::Pong => {
                if let Some(sent_at) = self.last_ping_sent.take() {
                    let rtt = u64::try_from(sent_at.elapsed().as_millis()).unwrap_or(u64::MAX);
                    tracing::debug!("RTT measurement: {}ms", rtt);
                    if let Err(e) = self.event_tx.send(NetworkEvent::Latency(rtt)).await {
                        error!("Failed to relay latency: {e}");
                    }
                }
            }
            _ => {
                if process_remote_message(msg, &self.event_tx).await.is_break() {
                    info!("Remote message processing signaled disconnect (e.g., Quit)");
                    self.cleanup_session();
                }
            }
        }
    }

    /// Relays an error message to the [`NetworkManager`].
    async fn send_error(&self, msg: String) {
        if let Err(send_err) = self.event_tx.send(NetworkEvent::Error(msg)).await {
            error!("Failed to relay error: {send_err}");
        }
    }
}

/// The main asynchronous entry point for the networking task.
///
/// This function initializes UDP and TCP sockets and enters a `select!` loop to
/// handle discovery, broadcasts, incoming connections, and messages.
///
/// # Errors
/// Returns an error if the initial socket setup fails.
async fn run_network_loop(
    mut cmd_rx: mpsc::Receiver<NetworkCommand>,
    event_tx: mpsc::Sender<NetworkEvent>,
) -> Result<()> {
    info!("Network loop starting");

    // Setup UDP Discovery Listener
    let udp_listener = {
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_reuse_address(true)?;
        #[cfg(not(windows))]
        socket.set_reuse_port(true)?;
        socket.set_nonblocking(true)?;
        socket.bind(&SocketAddr::from(([0, 0, 0, 0], UDP_PORT)).into())?;
        UdpSocket::from_std(socket.into())?
    };
    let mut udp_buf = [0u8; 1024];

    // Setup UDP Broadcast Socket
    let mut broadcast_interval = tokio::time::interval(Duration::from_secs(1));
    let mut interface_scan_interval = tokio::time::interval(Duration::from_secs(10));
    let udp_broadcast_socket = UdpSocket::bind("0.0.0.0:0").await?;
    udp_broadcast_socket.set_broadcast(true)?;

    // Heartbeat ticker
    let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(5));

    let mut state = NetworkLoopState::new(event_tx);
    let mut broadcast_targets = get_broadcast_targets();

    loop {
        tokio::select! {
            // 1. UDP Discovery
            res = udp_listener.recv_from(&mut udp_buf) => {
                state.handle_udp_discovery(res, &udp_buf).await;
            }

            // 2. UDP Broadcast
            _ = broadcast_interval.tick(), if state.hosting_info.is_some() => {
                for addr in &broadcast_targets {
                    state.handle_udp_broadcast(&udp_broadcast_socket, *addr).await;
                }
            }

            // 3. Topology Rescan
            _ = interface_scan_interval.tick() => {
                broadcast_targets = get_broadcast_targets();
            }

            // 4. TCP Accept (Hosting)
            accept_res = async {
                if let Some(ref l) = state.tcp_listener {
                    l.accept().await.ok()
                } else {
                    std::future::pending().await
                }
            }, if state.tcp_listener.is_some() && state.stream.is_none() => {
                state.handle_tcp_accept(accept_res).await;
            }

            // 4. Command Handling
            cmd_res = cmd_rx.recv() => {
                if !state.handle_command(cmd_res).await {
                    break;
                }
            }

            // 5. Heartbeat
            _ = heartbeat_interval.tick(), if state.stream.is_some() => {
                state.handle_heartbeat().await;
            }

            // 6. TCP Data Receiver
            res = async {
                if let Some(ref mut s) = state.stream {
                    // Apply a 15-second timeout to detect dead TCP connections
                    match tokio::time::timeout(Duration::from_secs(15), s.next()).await {
                        Ok(Some(Ok(bytes))) => Some(Ok(bytes)),
                        Ok(Some(Err(e))) => Some(Err(e)), // Framing/IO error
                        Ok(None) | Err(_) => None,        // Stream closed or timed out
                    }
                } else {
                    std::future::pending().await
                }
            }, if state.stream.is_some() => {
                state.handle_tcp_input(res).await;
            }
        }
    }
    info!("Network loop exited");
    Ok(())
}

/// Connects to a remote host and performs the initial handshake.
///
/// # Errors
/// Returns an error if the connection fails or the handshake is rejected.
async fn connect_to_host(
    addr: SocketAddr,
    player_name: String,
    event_tx: mpsc::Sender<NetworkEvent>,
) -> Result<Framed<TcpStream, LengthDelimitedCodec>> {
    let tcp_stream = tokio::time::timeout(
        Duration::from_millis(NETWORK_TIMEOUT_MS),
        TcpStream::connect(addr),
    )
    .await
    .map_err(|_| {
        NetworkError::Other(format!(
            "Connection to {addr} timed out after {NETWORK_TIMEOUT_MS}ms"
        ))
    })??;
    tcp_stream.set_nodelay(true)?;
    let mut framed = Framed::new(
        tcp_stream,
        LengthDelimitedCodec::builder()
            .max_frame_length(MAX_FRAME_SIZE)
            .new_codec(),
    );

    let role = ConnectionRole::Client { player_name };
    let result = perform_handshake(&mut framed, role).await?;

    if let Err(e) = event_tx
        .send(NetworkEvent::Ready {
            peer_name: result.peer_name,
            board_size: result.board_size,
            peer_player: result.peer_player,
        })
        .await
    {
        error!("Failed to relay connection ready event: {e}");
    }

    Ok(framed)
}

/// Dispatches a received [`RemoteMessage`] to the appropriate event handlers.
///
/// Returns `ControlFlow::Break` if the message signals a session termination.
async fn process_remote_message(
    msg: RemoteMessage,
    event_tx: &mpsc::Sender<NetworkEvent>,
) -> ControlFlow<()> {
    match msg {
        RemoteMessage::Move { col, state_hash } => {
            if let Err(e) = event_tx
                .send(NetworkEvent::MoveReceived { col, state_hash })
                .await
            {
                error!("Failed to relay move received event: {e}");
            }
        }
        RemoteMessage::Chat(m) => {
            let mut msg = m;
            truncate_string(&mut msg, MAX_CHAT_LEN);
            if let Err(e) = event_tx.send(NetworkEvent::ChatReceived(msg)).await {
                error!("Failed to relay chat received event: {e}");
            }
        }
        RemoteMessage::Quit => {
            if let Err(e) = event_tx
                .send(NetworkEvent::Disconnected("Opponent quit".to_string()))
                .await
            {
                error!("Failed to relay opponent quit disconnect: {e}");
            }
            return ControlFlow::Break(());
        }
        _ => {}
    }
    ControlFlow::Continue(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_message_serialization() {
        let msgs = vec![
            RemoteMessage::Handshake {
                version: 1,
                player_name: "Test".to_string(),
                board_size: (7, 6),
                local_player: Player::Red,
            },
            RemoteMessage::Move {
                col: 3,
                state_hash: 12345,
            },
            RemoteMessage::Chat("Hello".to_string()),
            RemoteMessage::Ping,
            RemoteMessage::Pong,
            RemoteMessage::Quit,
        ];
        for msg in msgs {
            let encoded = postcard::to_stdvec(&msg).expect("Failed to encode");
            let decoded: RemoteMessage = postcard::from_bytes(&encoded).expect("Failed to decode");
            match (&msg, &decoded) {
                (RemoteMessage::Move { col: c1, .. }, RemoteMessage::Move { col: c2, .. }) => {
                    assert_eq!(
                        c1, c2,
                        "Move column mismatch after serialization: sent {c1}, received {c2}"
                    );
                }
                (RemoteMessage::Chat(s1), RemoteMessage::Chat(s2)) => assert_eq!(
                    s1, s2,
                    "Chat message mismatch after serialization: sent '{s1}', received '{s2}'"
                ),
                (
                    RemoteMessage::Handshake {
                        local_player: p1, ..
                    },
                    RemoteMessage::Handshake {
                        local_player: p2, ..
                    },
                ) => assert_eq!(
                    p1, p2,
                    "Handshake player mismatch after serialization: sent {p1:?}, received {p2:?}"
                ),
                _ => {}
            }
        }
    }

    #[test]
    fn test_handshake_includes_host_player_bug() {
        let msg = RemoteMessage::Handshake {
            version: PROTOCOL_VERSION,
            player_name: "Host".to_string(),
            board_size: (7, 6),
            local_player: Player::Yellow,
        };
        let encoded = postcard::to_stdvec(&msg).unwrap();
        let decoded: RemoteMessage = postcard::from_bytes(&encoded).unwrap();
        if let RemoteMessage::Handshake { local_player, .. } = decoded {
            assert_eq!(
                local_player,
                Player::Yellow,
                "Handshake must preserve host player color"
            );
        } else {
            panic!("Decoded message is not a Handshake");
        }
    }

    #[tokio::test]
    async fn test_no_redundant_validation_bug() {
        let (tx, mut rx) = mpsc::channel(1);
        let msg = RemoteMessage::Move {
            col: 100, // Way out of bounds for any normal board
            state_hash: 42,
        };

        // Should relay regardless of any internal network loop column state
        let _ = process_remote_message(msg, &tx).await;

        let event = rx.try_recv().unwrap();
        if let NetworkEvent::MoveReceived { col, .. } = event {
            assert_eq!(
                col, 100,
                "Network layer should relay moves without validation (handled by game logic)"
            );
        } else {
            panic!("Expected MoveReceived event");
        }
    }

    #[test]
    fn test_discovery_addr_hybrid_bug() {
        // Source UDP address: ephemeral port 55555
        let src_addr: SocketAddr = "192.168.1.10:55555".parse().unwrap();
        // Reported TCP address: port 4444
        let mut info = DiscoveryInfo {
            instance_id: 1,
            local_name: "Host".to_string(),
            protocol_version: PROTOCOL_VERSION,
            addr: "0.0.0.0:4444".parse().unwrap(),
            board_size: (7, 6),
            host_player: Player::Red,
        };

        // Logic from network.rs loop:
        let mut final_src = src_addr;
        final_src.set_port(info.addr.port());
        info.addr = final_src;

        assert_eq!(
            info.addr.to_string(),
            "192.168.1.10:4444",
            "Discovery address must combine source IP with reported TCP port"
        );
    }

    #[tokio::test]
    async fn test_discovery_version_mismatch_bug() {
        let (event_tx, mut event_rx) = mpsc::channel(1);
        let state = NetworkLoopState::new(event_tx);

        let info = DiscoveryInfo {
            instance_id: 1,
            local_name: "Host".to_string(),
            protocol_version: PROTOCOL_VERSION + 1, // Wrong version
            addr: "127.0.0.1:4444".parse().unwrap(),
            board_size: (7, 6),
            host_player: Player::Red,
        };
        let data = postcard::to_stdvec(&info).unwrap();
        let src_addr: SocketAddr = "127.0.0.1:55555".parse().unwrap();

        state
            .handle_udp_discovery(Ok((data.len(), src_addr)), &data)
            .await;

        // Channel should be empty because packet was rejected
        assert!(
            event_rx.try_recv().is_err(),
            "Discovery packet with version mismatch should be rejected"
        );
    }
}
