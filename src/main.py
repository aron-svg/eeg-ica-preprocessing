import os

import pylsl.examples
from logger_init import logger
import json
import csv
from datetime import datetime
import pylsl
from pylsl import StreamInlet


import random
import time
from threading import Thread
from flask import Flask
from dash import Dash, html, dcc, Output, Input

# Simulate hardware device with random data
class SimulatedHardware:
    def __init__(self):
        self.data = 0
        self.running = True

    def read_data(self):
        """Simulate reading data from hardware."""
        self.data = random.randint(0, 100)

    def stop(self):
        self.running = False


# Flask server for Dash
server = Flask(__name__)

# Dash App
app = Dash(__name__, server=server)
app.title = "Hardware Data Viewer"

# Layout of the Dash web app
app.layout = html.Div([
    html.H1("Hardware Data Viewer", style={"textAlign": "center"}),
    html.Div(id="hardware-output", style={"fontSize": "24px", "textAlign": "center"}),
    dcc.Interval(id="update-interval", interval=1000, n_intervals=0),  # Update every second
])


# Simulated hardware instance
hardware = SimulatedHardware()


# Background thread to simulate hardware updates
def hardware_thread():
    while hardware.running:
        hardware.read_data()
        time.sleep(1)  # Simulate hardware data update interval


# Start hardware simulation thread
hardware_thread = Thread(target=hardware_thread)
hardware_thread.start()


# Callback to update the displayed hardware data
@app.callback(
    Output("hardware-output", "children"),
    Input("update-interval", "n_intervals")
)
def update_hardware_output(n_intervals):
    return f"Hardware Data: {hardware.data}"


# Run the app
if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    finally:
        hardware.stop()
        hardware_thread.join()


# if __name__ == "__main__":
#     logger.info("Starting the main process")
    
#     print("hello world")
    
#     pylsl.examples
#     StreamInlet.open_stream('type', 'EEG')
    
#     StreamInlet.close_stream('type', 'EEG')
#     pylsl.resolve_stream('type', 'EEG')
    
