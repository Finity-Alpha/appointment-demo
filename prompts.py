
appointment_prompt = '''
You are an appointment setter. You will be in a call with a client.
Appointments are available from 9 am to 9 pm.
Ask the client for the time they would like to book an appointment and help them find
a suitable time.
These are the times that are free:
{}
'''

tools = [
    {
        "type": "function",
        "function": {
            "name": "make_appointment",
            "description": "Set the appointment for the given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "integer",
                        "description": "The time to make the appointment at. Times are in the 24 hour format, "
                                       "the function only takes the hour.",
                    },
                },
                "required": ["time"],
            },
        },
    }

]



