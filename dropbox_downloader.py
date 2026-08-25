import sys
import urllib.request


def get_dropbox_direct_url(shared_url):
    if "dl=0" in shared_url:
        return shared_url.replace("dl=0", "dl=1")
    elif "dl=1" not in shared_url:
        if "?" in shared_url:
            return shared_url + "&dl=1"
        else:
            return shared_url + "?dl=1"
    return shared_url


def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, (read_so_far * 100) / total_size)
        sys.stdout.write(
            f"\rDownloading: {percent:.1f}% ({read_so_far / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading: {read_so_far / (1024 * 1024):.2f} MB (Total size unknown)")
        sys.stdout.flush()


def download_from_dropbox(dropbox_url, output_filename):
    direct_url = get_dropbox_direct_url(dropbox_url)
    print(f"[*] Original Link: {dropbox_url}")
    print(f"[*] Target Link  : {direct_url}")
    print(f"[*] Saving to    : {output_filename}")

    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
        urllib.request.install_opener(opener)

        print("[*] Starting download...")
        urllib.request.urlretrieve(direct_url, output_filename, download_progress)
        print("\n[+] Download completed successfully!")

    except Exception as e:
        print(f"\n[-] Error occurred during download: {e}")
        print("[!] Please check if the Dropbox link is public and still valid.")


if __name__ == "__main__":
    DROPBOX_LINK = input()
    SAVE_PATH = "./downloaded_file.zip"

    download_from_dropbox(DROPBOX_LINK, SAVE_PATH)
