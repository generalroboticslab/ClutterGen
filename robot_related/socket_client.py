import socket
from sshkeyboard import listen_keyboard

class SocketClient:
    def __init__(self, server_ip="10.197.245.55", port=5050):
        self.HEADER = 64
        self.PORT = port
        self.SERVER = server_ip
        self.ADDR = (self.SERVER, self.PORT)
        self.FORMAT = 'utf-8'
        self.DISCONNECT_MESSAGE = "!DISCONNECT"

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_server()


    def connect_to_server(self):
        try:
            self.client.connect(self.ADDR)
            print(f"[CONNECTED] Connected to {self.ADDR}")
        except Exception as e:
            print(f"[ERROR] {e}")
            print("[ERROR] Could not connect to the server; Please check the server is start or not and IP address is correct.")
            exit(1)


    def send_message(self, raw_list):
        if isinstance(raw_list, str):
            raw_list = [raw_list, None]
        elif isinstance(raw_list, list):
            raw_list = raw_list
        else:
            raise ValueError(f"raw_list should be a list [command, value] or pure str, but got {type(raw_list)} type.")
        assert len(raw_list) == 2, f"raw_list should have length 2, but got {len(raw_list)}."
        msg = str(raw_list)
        message = msg.encode(self.FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(self.FORMAT)
        send_length += b' ' * (self.HEADER - len(send_length))
        self.client.send(send_length)
        self.client.send(message)


    def receive_message(self):
        # self.client.settimeout(None)  # Timeout after 5 seconds; None is blocking mode
        header = self.client.recv(self.HEADER).decode(self.FORMAT)
        if header:
            msg_length = int(header.strip())
            message = self.client.recv(msg_length).decode(self.FORMAT)
            return message


    def disconnect(self):
        self.send_message([self.DISCONNECT_MESSAGE, None])
        self.client.close()
        print("[DISCONNECTED] Disconnected from the server")

    
    def keyboard_control(self, key):
        key = str(key)
        if key == "Key.esc":
            self.disconnect()
            return False
        self.send_message(key)
        return True


    def keyboard_start(self):
        try:
            with listen_keyboard(on_press=self.keyboard_control) as listener:
                listener.join()
        except Exception as e:
            print("Exiting...")
            pass


if __name__ == "__main__":
    client = SocketClient()  # Use the appropriate server IP and port
    # Example usage:
    client.send_message(["Hello Server!", None])
    response = client.receive_message()  # Assuming the server sends a response
    client.keyboard_start()
    client.disconnect()