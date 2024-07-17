# Voice Appointment Bot using OVC

A simple appointment booking voice AI bot built using OpenVoiceChat.

A csv file contains the timings and bookings, function calling is used to 
check the availability of the time slot and book the appointment.

Models used are whisper-base, openai's gpt and xtts_v2.
The code takes 3GB of gpu VRAM.

The [all_in_one.py](all_in_one.py) file contains the entire code in one py 
file, including the prompts. The file is 200 lines long, this is to 
illustrate how easy it has become to make a voice AI bot using OVC.

