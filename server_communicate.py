import subprocess
import os
import paramiko
from scp import SCPClient

def upload_file(local_path, server_path, server_address, username, password):
    """
    Uploads a local file to a remote server using SCP.

    :param local_path: Local path of the file to upload.
    :param server_path: Remote path where the file should be uploaded.
    :param server_address: Address of the remote server.
    :param username: SSH username for authentication.
    :param password: SSH password for authentication.
    """
    # Establish an SSH connection
    with paramiko.Transport((server_address, 22)) as transport:
        transport.connect(username=username, password=password)

        # Use SSHClient to create the remote folder if it doesn't exist
        with paramiko.SSHClient() as ssh:
            ssh._transport = transport  # Assign the existing transport to the SSHClient
            ssh.exec_command(f'mkdir -p {server_path}')  # Create the remote folder

            # Create an SCP client
            with SCPClient(transport) as scp:
                # Upload the file to the remote folder
                scp.put(local_path, recursive=True, remote_path=server_path)
    
    print(f"Uploaded:\n{local_path}\nto\n{server_path}.")


def download_file(server_path, local_path, server_address, username, password):
    """
    Downloads a file from a remote server to a local path using SCP.

    :param local_path: Local path where the file should be saved.
    :param server_path: Remote path of the file on the server.
    :param server_address: Address of the remote server.
    :param username: SSH username for authentication.
    :param password: SSH password for authentication.
    """

    # Establish an SSH connection
    with paramiko.Transport((server_address, 22)) as transport:
        transport.connect(username=username, password=password)

        # Create an SCP client
        with SCPClient(transport) as scp:
            subprocess.run(f'mkdir -p {local_path}', shell=True)  # Create the local folder if it doesn't exist
            # Download the file from the remote path to the local path
            scp.get(server_path, recursive=True, local_path=local_path)

    print(f"Downloaded:\n{server_path}\nto\n{local_path}.")


if __name__=="__main__":
    import argparse
    from str2bool import str2bool as strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--relative_file_path", type=str, required=True)
    parser.add_argument("-s", "--send", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("-ip", "--server", type=str, default="00")
    args = parser.parse_args()

    # Example usage:
    if args.send:
        assert os.path.exists(args.relative_file_path), f"File does not exist: {args.relative_file_path}"
    local_project_path = "/home/grl/Working/projects/RoboSensai"
    server_project_path = "/home/yj208/Working/RoboSensai"
    local_abs_file_path = os.path.join(local_project_path, args.relative_file_path)
    server_abs_file_path = os.path.join(server_project_path, args.relative_file_path)
    server_address = f"bc298-cmp-{args.server}.egr.duke.edu"
    username = "yj208"
    password = "Jys11053032!"

    if args.send:
        upload_file(local_abs_file_path, os.path.dirname(server_abs_file_path), server_address, username, password)
    else:
        # Clear the local file before downloading
        if os.path.exists(local_abs_file_path):
            key = input(f"Press Enter to delete {local_abs_file_path} and download the file from the server...")
            if key == "": os.remove(local_abs_file_path)
            else: raise Exception("Invalid input.")

        download_file(server_abs_file_path, os.path.dirname(local_abs_file_path), server_address, username, password)
