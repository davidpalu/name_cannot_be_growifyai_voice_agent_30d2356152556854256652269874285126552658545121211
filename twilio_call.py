from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    raise RuntimeError("Twilio credentials missing in .env")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def call_number(to_number: str):
    """
    Call a phone number FROM your Twilio number
    """
    call = client.calls.create(
        to=to_number,
        from_=TWILIO_PHONE_NUMBER,
        twiml=f"""
        <Response>
            <Start>
                <Stream url="wss://underclad-travis-overminutely.ngrok-free.dev/media" />
            </Start>
            <Say>Start speaking</Say>
            <Pause length="60"/>
        </Response>
        """
    )

    print("📞 Call started")
    print("Call SID:", call.sid)
    return call.sid


if __name__ == "__main__":
    call_number("+918309507529")
