use crate::{Error, mocks};
use commonware_utils::{channel::mpsc, sync::Mutex};
use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    ops::Range,
    sync::Arc,
};

/// Range of ephemeral ports assigned to dialers.
const EPHEMERAL_PORT_RANGE: Range<u16> = 32768..61000;

/// Implementation of [crate::Sink] for a deterministic [Network].
pub type Sink = mocks::Sink;

/// Implementation of [crate::Stream] for a deterministic [Network].
pub type Stream = mocks::Stream;

/// Implementation of [crate::Listener] for a deterministic [Network].
///
/// Dropping it frees its address for another [crate::Network::bind].
pub struct Listener {
    address: SocketAddr,
    listener: mpsc::UnboundedReceiver<(SocketAddr, mocks::Sink, mocks::Stream)>,
    listeners: Listeners,
}

impl Drop for Listener {
    fn drop(&mut self) {
        // Closing the receiver closes its sender, so only this listener's entry
        // is removed: one a later listener made would still be open.
        self.listener.close();
        let mut listeners = self.listeners.lock();
        if listeners
            .get(&self.address)
            .is_some_and(mpsc::UnboundedSender::is_closed)
        {
            listeners.remove(&self.address);
        }
    }
}

impl crate::Listener for Listener {
    type Sink = Sink;
    type Stream = Stream;

    async fn accept(&mut self) -> Result<(SocketAddr, Self::Sink, Self::Stream), Error> {
        let (socket, sender, receiver) = self.listener.recv().await.ok_or(Error::ReadFailed)?;
        Ok((socket, sender, receiver))
    }

    fn local_addr(&self) -> Result<SocketAddr, std::io::Error> {
        Ok(self.address)
    }
}

type Dialable = mpsc::UnboundedSender<(
    SocketAddr,
    mocks::Sink,   // Listener -> Dialer
    mocks::Stream, // Dialer -> Listener
)>;

type Listeners = Arc<Mutex<HashMap<SocketAddr, Dialable>>>;

/// Deterministic implementation of [crate::Network].
///
/// When a dialer connects to a listener, the listener is given the next ephemeral
/// port from the range `32768..61000`, starting again from its beginning once it
/// is used up: a dialer's port only labels the connection, which nothing reaches
/// by it. To keep things simple, it is not possible to bind to an ephemeral port.
/// A dropped listener frees its address.
#[derive(Clone)]
pub struct Network {
    ephemeral: Arc<Mutex<u16>>,
    listeners: Listeners,
}

impl Default for Network {
    fn default() -> Self {
        Self {
            ephemeral: Arc::new(Mutex::new(EPHEMERAL_PORT_RANGE.start)),
            listeners: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl crate::Network for Network {
    type Listener = Listener;

    async fn bind(&self, socket: SocketAddr) -> Result<Self::Listener, Error> {
        // If the IP is localhost, ensure the port is not in the ephemeral range
        // so that it can be used for binding in the dial method
        if socket.ip() == IpAddr::V4(Ipv4Addr::LOCALHOST)
            && EPHEMERAL_PORT_RANGE.contains(&socket.port())
        {
            return Err(Error::BindFailed);
        }

        // Ensure the port is not already bound
        let mut listeners = self.listeners.lock();
        if listeners
            .get(&socket)
            .is_some_and(|sender| !sender.is_closed())
        {
            return Err(Error::BindFailed);
        }

        // Bind the socket
        let (sender, receiver) = mpsc::unbounded_channel();
        listeners.insert(socket, sender);
        Ok(Listener {
            address: socket,
            listener: receiver,
            listeners: self.listeners.clone(),
        })
    }

    async fn dial(&self, socket: SocketAddr) -> Result<(Sink, Stream), Error> {
        // Assign dialer a port from the ephemeral range
        let dialer = {
            let mut ephemeral = self.ephemeral.lock();
            let dialer = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), *ephemeral);
            *ephemeral = if *ephemeral + 1 < EPHEMERAL_PORT_RANGE.end {
                *ephemeral + 1
            } else {
                EPHEMERAL_PORT_RANGE.start
            };
            dialer
        };

        // Get listener
        let sender = {
            let listeners = self.listeners.lock();
            let sender = listeners.get(&socket).ok_or(Error::ConnectionFailed)?;
            sender.clone()
        };

        // Construct connection
        let (dialer_sender, dialer_receiver) = mocks::Channel::init();
        let (listener_sender, listener_receiver) = mocks::Channel::init();
        sender
            .send((dialer, dialer_sender, listener_receiver))
            .map_err(|_| Error::ConnectionFailed)?;
        Ok((listener_sender, dialer_receiver))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Clock, Error, Listener as _, Network as _, Runner, Spawner,
        network::{deterministic as DeterministicNetwork, tests},
    };
    use commonware_macros::test_group;
    use rstest::rstest;
    use std::net::SocketAddr;

    #[rstest]
    #[case::tokio(crate::tokio::Runner::default())]
    #[cfg_attr(target_os = "linux", case::iouring(crate::iouring::Runner::default()))]
    fn test_trait<R: Runner>(#[case] runner: R)
    where
        R::Context: Spawner + Clock,
    {
        runner.start(|context| async move {
            tests::test_network_trait(context, DeterministicNetwork::Network::default).await;
        });
    }

    #[test]
    fn test_dropped_listener_frees_its_address() {
        crate::deterministic::Runner::default().start(|_| async move {
            let network = DeterministicNetwork::Network::default();
            let address = SocketAddr::from(([127, 0, 0, 1], 3000));
            let listener = network.bind(address).await.unwrap();
            assert!(matches!(
                network.bind(address).await,
                Err(Error::BindFailed)
            ));
            drop(listener);
            let _listener = network.bind(address).await.unwrap();
            assert!(network.dial(address).await.is_ok());
        });
    }

    #[test]
    fn test_ephemeral_ports_wrap() {
        crate::deterministic::Runner::default().start(|_| async move {
            let network = DeterministicNetwork::Network::default();
            let address = SocketAddr::from(([127, 0, 0, 1], 3000));
            let mut listener = network.bind(address).await.unwrap();
            let range = super::EPHEMERAL_PORT_RANGE;
            for _ in range.clone() {
                let (_sink, _stream) = network.dial(address).await.unwrap();
                listener.accept().await.unwrap();
            }
            let _connection = network.dial(address).await.unwrap();
            let (dialer, _, _) = listener.accept().await.unwrap();
            assert_eq!(dialer.port(), range.start);
        });
    }

    #[rstest]
    #[case::tokio(crate::tokio::Runner::default())]
    #[cfg_attr(target_os = "linux", case::iouring(crate::iouring::Runner::default()))]
    #[test_group("slow")]
    fn test_stress_trait<R: Runner>(#[case] runner: R)
    where
        R::Context: Spawner + Clock,
    {
        runner.start(|context| async move {
            tests::stress_test_network_trait(context, DeterministicNetwork::Network::default).await;
        });
    }
}
