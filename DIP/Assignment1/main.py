from email.mime import image
from band import band
from channels import channels
from display import display_image
from operations import operations

def main():
    images=['1.jpg','2.jpg','Beach (13).JPEG']
    display_image(images)
    band(images)
    channels(images)
    operations(images)

if __name__ == "__main__":
    main()