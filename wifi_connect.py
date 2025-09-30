import subprocess
import re
import time
import os

def get_available_ssids():
    """
    Scan for available WiFi SSIDs using iwlist.
    """
    try:
        scan_output = subprocess.check_output(['sudo', 'iwlist', 'wlan0', 'scan']).decode('utf-8')
        ssids = re.findall(r'ESSID:"(.*?)"', scan_output)
        return set(ssids)
    except Exception as e:
        print(f"Error scanning WiFi: {e}")
        return set()

def is_connected():
    """
    Check if wlan0 is connected to a WiFi network.
    """
    try:
        status = subprocess.check_output(['iwconfig', 'wlan0']).decode('utf-8')
        return 'ESSID:off/any' not in status
    except Exception as e:
        print(f"Error checking connection: {e}")
        return False

def add_network_to_wpa(ssid, password):
    """
    Add the network configuration to wpa_supplicant.conf if not already present.
    Handles file creation if it doesn't exist.
    """
    conf_path = '/etc/wpa_supplicant/wpa_supplicant.conf'
    network_block = f'\nnetwork={{\n    ssid="{ssid}"\n    psk="{password}"\n}}\n'
    header = """ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

"""
    
    try:
        if os.path.exists(conf_path):
            with open(conf_path, 'r') as f:
                content = f.read()
        else:
            content = ''
            with open(conf_path, 'w') as f:
                f.write(header)
            os.chmod(conf_path, 0o600)  # Secure permissions
        
        if f'ssid="{ssid}"' in content:
            print(f"Network {ssid} already configured in {conf_path}")
            return True
        
        with open(conf_path, 'a') as f:
            f.write(network_block)
        
        print(f"Added network {ssid} to {conf_path}")
        return True
    except Exception as e:
        print(f"Error updating {conf_path}: {e}")
        return False

def setup_interfaces():
    """
    Add wlan0 configuration to /etc/network/interfaces if not present.
    """
    interfaces_path = '/etc/network/interfaces'
    stanza = """
allow-hotplug wlan0
auto wlan0
iface wlan0 inet dhcp
    wpa-conf /etc/wpa_supplicant/wpa_supplicant.conf
"""
    
    try:
        with open(interfaces_path, 'r') as f:
            content = f.read()
        
        if 'iface wlan0' in content:
            print("wlan0 already configured in /etc/network/interfaces")
            return True
        
        with open(interfaces_path, 'a') as f:
            f.write(stanza)
        
        print("Added wlan0 configuration to /etc/network/interfaces")
        return True
    except Exception as e:
        print(f"Error updating /etc/network/interfaces: {e}")
        return False

def reconfigure_wpa():
    """
    Reconfigure the interface by bringing it down and up.
    """
    try:
        subprocess.call(['sudo', 'ifdown', '--force', 'wlan0'])
        time.sleep(5)
        subprocess.call(['sudo', 'ifup', 'wlan0'])
        time.sleep(15)  # Give more time for connection attempt
        return True
    except Exception as e:
        print(f"Error reconfiguring interface: {e}")
        return False

def main():
    if is_connected():
        print("Already connected to a WiFi network. No action needed.")
        return
    
    available_ssids = get_available_ssids()
    preferred_ssids = ["Victus", "Pixel 7", "Victus 5G"]  # Added "Victus 5G" based on screenshot
    password = "68986898"
    connected = False
    
    for ssid in preferred_ssids:
        if ssid in available_ssids:
            print(f"Preferred SSID '{ssid}' found. Attempting to connect...")
            if add_network_to_wpa(ssid, password):
                if setup_interfaces():
                    if reconfigure_wpa():
                        if is_connected():
                            print(f"Successfully connected to '{ssid}'.")
                            connected = True
                            break
                        else:
                            print(f"Failed to connect to '{ssid}' after reconfiguration.")
                    else:
                        print("Failed to update /etc/network/interfaces.")
                else:
                    print("Failed to reconfigure the interface.")
            else:
                print(f"Failed to add '{ssid}' to configuration.")
    
    if not connected:
        print("No preferred WiFi networks found or connection attempts failed.")

if __name__ == "__main__":
    main()