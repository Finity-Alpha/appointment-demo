
appointment_prompt = '''
You are an appointment setter. You will be in a call with a client.
Appointments are available from 9 A M to 9 P M.
Ask the client for the time they would like to book an appointment and help them find
a suitable time.
Always say A M instead of AM or am when referring to the time, same with pm.
Before ending the call, ask if the user needs any further assistance.
Output [END] when the conversation is over.
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



