from stream import app
import socketio
from waitress import serve
import socket

# Get local ip address
def get_local_ip():
    # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # try to connect to google public dns
    try:
        s.connect(('8.8.8.8', 1))
        # get local ip address
        local_ip = s.getsockname()[0]
    # if no internet connection, use localhost
    except Exception:
        local_ip = '127.0.0.1'
    # close socket
    finally:
        s.close()
    return local_ip

# Get local ip address
ip_address = get_local_ip()
# Number of threads
thread = 12
# SocketIO
sio = socketio.Server()
# Wrap Flask application with SocketIO's middleware
app_server = socketio.WSGIApp(sio, app)

# SocketIO event handler
if __name__ == '__main__':
    print("Server running at: http://"+ip_address+":8080 or http://localhost:8080,\nThreads: "+str(thread))
    # Run server
    serve(app_server, host='0.0.0.0', port=8080, url_scheme='http', threads=thread)