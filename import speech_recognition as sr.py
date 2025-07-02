import speech_recognition as sr

# Function to simulate wheelchair commands and print to the console
def send_command_to_wheelchair(command):
    if command == "forward":
        print("Command: Moving forward...")
    elif command == "backward":
        print("Command: Moving backward...")
    elif command == "left":
        print("Command: Turning left...")
    elif command == "right":
        print("Command: Turning right...")
    elif command == "break" or command == "stop":
        print("Command: Stopping wheelchair...")
    else:
        print("Command not recognized!")

# Function to recognize voice commands
def recognize_voice_command():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)

    try:
        # Use Google's speech recognition service
        command = recognizer.recognize_google(audio).lower()
        print(f"Recognized command: {command}")
        send_command_to_wheelchair(command)

    except sr.UnknownValueError:
        print("Sorry, I did not understand the command.")
    except sr.RequestError:
        print("Error with the speech recognition service.")

if __name__ == "__main__":  # Corrected here
    
    while True:
        recognize_voice_command()
