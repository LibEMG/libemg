from multiprocessing import Value
import os
import requests
import shutil
import json
from semantic_version import Version
from platform import system
import re


# this is responsible for receiving the data
class SiFiBridge:
    def __init__(self, config, version, other, bridge_version: str | None = None):
        self.version = version

        pltfm = system()
        executable = f"sifi_bridge%s-{pltfm.lower()}" + (
            ".exe" if pltfm == "Windows" else ""
        )

        if bridge_version is None:
            # Find the latest upstream version
            try:
                releases = requests.get(
                    "https://api.github.com/repos/sifilabs/sifi-bridge-pub/releases",
                    timeout=5,
                ).json()
                bridge_version = str(
                    max([Version(release["tag_name"]) for release in releases])
                )
            except Exception:
                # Probably some network error, so try to find an existing version
                # Expected to find sifi_bridge-V.V.V-platform in the current directory
                for file in os.listdir():
                    if not file.startswith("sifi_bridge"):
                        continue
                    bridge_version = file.split("-")[1].replace(".exe", "")
                if bridge_version is None:
                    raise ValueError(
                        "Could not fetch from upstream nor find a version of sifi_bridge to use in the current directory."
                    )

        executable = executable % ("-" + bridge_version)

        if executable not in os.listdir():
            ext = ".zip" if pltfm == "Windows" else ".tar.gz"
            arch = None
            if pltfm == "Linux":
                arch = "x86_64-unknown-linux-gnu"
                print(
                    "Please run <chmod +x sifi_bridge> in the terminal to indicate this is an executable file! You only need to do this once."
                )
            elif pltfm == "Darwin":
                arch = "aarch64-apple-darwin"
            elif pltfm == "Windows":
                arch = "x86_64-pc-windows-gnu"

            # Get Github releases
            releases = requests.get(
                "https://api.github.com/repos/sifilabs/sifi-bridge-pub/releases",
                timeout=5,
            ).json()

            # Extract the release matching the requested version
            release_idx = [release["tag_name"] for release in releases].index(
                bridge_version
            )
            assets = releases[release_idx]["assets"]

            # Find the asset that matches the architecture
            archive_url = None
            for asset in assets:
                asset_name = asset["name"]
                if arch not in asset_name:
                    continue
                archive_url = asset["browser_download_url"]
            if not archive_url:
                ValueError(f"No upstream version found for {executable}")
            print(f"Fetching sifi_bridge from {archive_url}")

            # Fetch and write to disk as a zip file
            r = requests.get(archive_url)
            zip_path = "sifi_bridge" + ext
            with open(zip_path, "wb") as file:
                file.write(r.content)

            # Unpack & delete the archive
            shutil.unpack_archive(zip_path, "./")
            os.remove(zip_path)
            extracted_path = f"sifi_bridge-{bridge_version}-{arch}/"
            for file in os.listdir(extracted_path):
                if not file.startswith("sifi_bridge"):
                    continue
                shutil.move(extracted_path + file, f"./{executable}")
            shutil.rmtree(extracted_path)

        self.proc = subprocess.Popen(
            ["./" + executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        assert self.proc is not None

        self.config = config
        self.other = other
        self.emg_handlers = []
        self.imu_handlers = []
        self.other_handlers = []

    def connect(self):
        connected = False
        while not connected:
            name = "-c BioPoint_v" + str(self.version) + "\n"
            self.proc.stdin.write(bytes(name, "utf-8"))
            self.proc.stdin.flush()

            ret = self.proc.stdout.readline().decode()

            dat = json.loads(ret)

            if dat["connected"] == 1:
                connected = True
            else:
                print("Could not connect. Retrying.")
        # Setup channels
        self.proc.stdin.write(self.config)
        self.proc.stdin.flush()
        print("Connected and sent configs to bridge.")

    def add_emg_handler(self, closure):
        self.emg_handlers.append(closure)

    def add_imu_handler(self, closure):
        self.imu_handlers.append(closure)

    def add_other_handler(self, closure):
        self.other_handlers.append(closure)

    def run(self):
        self.proc.stdin.write(b"-cmd 1\n")
        self.proc.stdin.flush()
        self.proc.stdin.write(b"-cmd 0\n")
        self.proc.stdin.flush()
        packet = np.zeros((14, 8))
        while True:
            data_arr_as_json = self.proc.stdout.readline().decode()
            if data_arr_as_json == "" or data_arr_as_json.startswith("sending cmd"):
                continue
            data_arr_as_json = json.loads(data_arr_as_json)
            if "data" in list(data_arr_as_json.keys()):
                if "emg0" in list(data_arr_as_json["data"].keys()):
                    for c in range(packet.shape[1]):
                        packet[:, c] = data_arr_as_json["data"]["emg" + str(c)]
                    for s in range(packet.shape[0]):
                        for h in self.emg_handlers:
                            h(packet[s, :].tolist())
                elif "emg" in list(
                    data_arr_as_json["data"].keys()
                ):  # This is the biopoint emg
                    emg = data_arr_as_json["data"]["emg"]
                    for e in emg:
                        if not self.other:
                            self.emg_handlers[0]([e])
                        else:
                            self.other_handlers[0]("EMG-bio", [e])
                if "acc_x" in list(data_arr_as_json["data"].keys()):
                    accel = np.transpose(
                        np.vstack(
                            [
                                data_arr_as_json["data"]["acc_x"],
                                data_arr_as_json["data"]["acc_y"],
                                data_arr_as_json["data"]["acc_z"],
                            ]
                        )
                    )
                    quat = np.transpose(
                        np.vstack(
                            [
                                data_arr_as_json["data"]["w"],
                                data_arr_as_json["data"]["x"],
                                data_arr_as_json["data"]["y"],
                                data_arr_as_json["data"]["z"],
                            ]
                        )
                    )
                    imu = np.hstack((accel, quat))
                    for i in imu:
                        if not self.other:
                            self.imu_handlers[0](i)
                        else:
                            self.other_handlers[0]("IMU-bio", i)
                if "eda" in list(data_arr_as_json["data"].keys()):
                    eda = data_arr_as_json["data"]["eda"]
                    for e in eda:
                        self.other_handlers[0]("EDA-bio", [e])
                if "b" in list(data_arr_as_json["data"].keys()):
                    sizes = [
                        len(data_arr_as_json["data"]["b"]),
                        len(data_arr_as_json["data"]["g"]),
                        len(data_arr_as_json["data"]["r"]),
                        len(data_arr_as_json["data"]["ir"]),
                    ]
                    ppg = np.transpose(
                        np.vstack(
                            [
                                data_arr_as_json["data"]["b"][0 : min(sizes)],
                                data_arr_as_json["data"]["g"][0 : min(sizes)],
                                data_arr_as_json["data"]["r"][0 : min(sizes)],
                                data_arr_as_json["data"]["ir"][0 : min(sizes)],
                            ]
                        )
                    )
                    for p in ppg:
                        self.other_handlers[0]("PPG-bio", p)

    def close(self):
        self.proc.stdin.write(b"-cmd 1\n")
        self.proc.stdin.flush()
        return

    def turnoff(self):
        self.proc.stdin.write(b"-cmd 13\n")
        self.proc.stdin.flush()
        return

    def __del__(self):
        self.proc.stdin.write(b"-d -q\n")


import subprocess
import json
import numpy as np
import socket
import pickle


class SiFiBridgeStreamer:
    def __init__(
        self,
        ip,
        port,
        version="1_2",
        ecg=False,
        emg=True,
        eda=False,
        imu=False,
        ppg=False,
        notch_on=True,
        notch_freq=60,
        emgfir_on=True,
        emg_fir=[20, 450],
        other=False,
    ):
        # notch_on refers to EMG notch filter
        # notch_freq refers to frequency cutoff of notch filter
        #
        self.other = other
        self.version = version
        self.ip = ip
        self.port = port
        self.config = "-s ch %s,%s,%s,%s,%s " % (
            str(int(ecg)),
            str(int(emg)),
            str(int(eda)),
            str(int(imu)),
            str(int(ppg)),
        )
        if notch_on or emgfir_on:
            self.config += "enable_filters 1 "
            emg_cfg = "emg_cfg %s,%s,%s "
            emg_notch = "0"
            if notch_on:
                emg_notch = str(notch_freq)
            self.config += emg_cfg % (emg_fir[1], emg_fir[0], emg_notch)
        else:
            self.config += "enable_filters 0 "

        print(self.config)
        self.config = bytes(self.config + "\n", "UTF-8")

    def start_stream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        b = SiFiBridge(self.config, self.version, self.other)
        b.connect()

        def write_emg(emg):
            data_arr = pickle.dumps(list(emg))
            sock.sendto(data_arr, (self.ip, self.port))

        b.add_emg_handler(write_emg)

        def write_imu(imu):
            imu_list = ["IMU", imu]
            data_arr = pickle.dumps(list(imu_list))
            sock.sendto(data_arr, (self.ip, self.port))

        b.add_imu_handler(write_imu)

        def write_other(other, data):
            other_list = [other, data]
            data_arr = pickle.dumps(list(other_list))
            sock.sendto(data_arr, (self.ip, self.port))

        b.add_other_handler(write_other)

        while True:
            try:
                b.run()
            except Exception as e:
                print("Error Occured: " + str(e))
                continue
