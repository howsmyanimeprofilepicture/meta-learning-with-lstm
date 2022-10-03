import os
from google_drive_downloader import GoogleDriveDownloader as gdd


def main():
    """
    Need to download the Omniglot dataset -- DON'T MODIFY THIS CELL

    """
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )

    assert os.path.isdir("./omniglot_resized")


if __name__ == "__main__":
    main()
