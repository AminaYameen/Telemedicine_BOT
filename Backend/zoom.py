import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")

def get_access_token():
    url = "https://zoom.us/oauth/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "grant_type": "account_credentials",
        "account_id": os.getenv("ZOOM_ACCOUNT_ID")
    }
    auth = (os.getenv("ZOOM_CLIENT_ID"), os.getenv("ZOOM_CLIENT_SECRET"))
    
    response = requests.post(url, headers=headers, data=payload, auth=auth)
    try:
        return response.json()["access_token"]
    except KeyError:
        print("❌ Error fetching token:")
        print(response.status_code, response.text)
        return None

def getUsers():
    """Fetch Zoom user information"""
    headers = {
        'authorization': f'Bearer {get_access_token()}',
        'content-type': 'application/json'
    }
    r = requests.get('https://api.zoom.us/v2/users/', headers=headers)
    print("\nFetching Zoom user information...\n")
    print(r.text)

def getMeetingParticipants():
    """Fetch participants of a live Zoom meeting"""
    headers = {
        'authorization': f'Bearer {get_access_token()}',
        'content-type': 'application/json'
    }
    r = requests.get(
        f'https://api.zoom.us/v2/metrics/meetings/participants',
        headers=headers
    )
    print("\nFetching Zoom meeting participants...\n")
    print(r.text)

# Meeting configuration
meetingdetails = {
    "topic": "Telemedicine",
    "type": 2,
    "start_time": "2025-05-14T10:21:57",
    "duration": "45",
    "timezone": "Asia/Karachi",
    "agenda": "test",
    "recurrence": {
        "type": 1,
        "repeat_interval": 1
    },
    "settings": {
        "host_video": True,
        "participant_video": True,
        "join_before_host": False,
        "mute_upon_entry": False,
        "watermark": True,
        "audio": "voip",
        "auto_recording": "cloud"
    }
}

def createMeeting():
    """Create a new Zoom meeting and show key details"""
    token = get_access_token()
    if not token:
        print("❌ Cannot create meeting without access token.")
        return

    headers = {
        'authorization': f'Bearer {token}',
        'content-type': 'application/json'
    }

    print("\nCreating Zoom meeting...\n")
    r = requests.post(
        'https://api.zoom.us/v2/users/me/meetings',
        headers=headers,
        data=json.dumps(meetingdetails)
    )

    if r.status_code == 201:
        meeting = r.json()
        print("✅ Zoom meeting created successfully!\n")
        print(f"🆔 Meeting ID: {meeting['id']}")
        print(f"📝 Topic: {meeting['topic']}")
        print(f"🕒 Start Time: {meeting['start_time']}")
        print(f"🌍 Timezone: {meeting['timezone']}")
        print(f"🔑 Password: {meeting['password']}")
        print(f"🔗 Join URL (for participants): {meeting['join_url']}")
        print(f"🧑‍💼 Start URL (for host): {meeting['start_url']}")
    else:
        print("❌ Failed to create Zoom meeting:")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text}")

# Example of function calls
getUsers()
createMeeting()
