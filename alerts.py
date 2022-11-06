import config 
from twilio.rest import Client  
import time


account_sid = config.account_sid
auth_token = config.auth_token
twilio_number = config.twilio_number
my_phone_number = "5107388942"
client = Client(account_sid, auth_token)

time.sleep(5)

message = client.messages.create(
    body = "Movement was detected, Check Your Stream",
    from_= twilio_number,
    to = my_phone_number
)
print(message.body)

time.sleep(2)

message1 = client.messages.create(
    body = "Rayyan is at your front door",
    from_= twilio_number,
    to = my_phone_number
)
print(message1.body)

time.sleep(2)

message2 = client.messages.create(
    body = "Someone may be armed outside your door",
    from_= twilio_number,
    to = my_phone_number
)


print(message2.body)
