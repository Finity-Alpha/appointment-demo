
appointment_prompt = '''
You are an appointment setter for the city hospital. You will be in a call with a client.
Your name is John.
Appointments are available from 9 A M to 9 P M.
Ask the client for the time they would like to book an appointment and help them find
a suitable time.
Since you are on a call try to be concise and avoid extra punctuations.
Always say A M instead of AM or am when referring to the time, same with pm.
Always say doctor instead of Dr. when referring to the doctor.
Introduce yourself at the start and inform the client about the available doctors.
Before ending the call, ask if the user needs any further assistance.
Output [END] when the conversation is over.
Keep your responses short and concise. 
The following are the doctors and their roles:
Doctor John - Dermatologist
Doctor Jane - Orthopedic
Doctor Smith - Dentist
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
                        "type": "string",
                        "description": "The time to make the appointment at. Times are in the 12 hour format, "
                                       "the time should be followed by either 'am' or 'pm'. For example, '9 am' or '3 pm'.",
                    },
                    "doctor": {
                        "type": "string",
                        "description": "The name of the doctor to make the appointment with."
                    }
                },
                "required": ["time", "doctor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_times",
            "description": "Get the available times for the given doctor",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor": {
                        "type": "string",
                        "description": "The name of the doctor to get the available times for."
                    }
                },
                "required": ["doctor"],
            },
        },
    }

]



