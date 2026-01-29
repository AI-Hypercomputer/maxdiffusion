import zmq
import pickle

def send_pyobj(socket, obj, identity=None):
    """Sends a Python object via ZMQ (pickle)."""
    data = pickle.dumps(obj)
    if identity:
        socket.send_multipart([identity, b"", data])
    else:
        socket.send(data)

def recv_pyobj(socket):
    """Receives a Python object via ZMQ (pickle)."""
    parts = socket.recv_multipart()
    if len(parts) >= 2 and parts[1] == b"":
        # Router pattern: [Identity, Delimiter, Data]
        identity = parts[0]
        data = parts[2]
        return identity, pickle.loads(data)
    else:
        # Dealer/Simple pattern
        data = parts[0]
        return pickle.loads(data)
