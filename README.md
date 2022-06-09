## JitterNot: ABR for VCAs with DNN and MPC

JitterNot is an innovative ABR approach for real-time video conferencing with the fusion of DL and model-predictive control methods. JitterNot improves upon the existing ABR approaches for VCAs as they use fixed control rules or heuristics, so, suffer from the limitation of not taking the deployment environment and network conditions into consideration. The salient features of our closed-loop control system are: (i) an LSTM model to guide the adaptive bitrate algorithm, (ii) an ABR algorithm, that utilizes an MPC controller, throughput prediction, and contextual information like Quantization Parameter (QP), and frame jitter.

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/02e6a8e9bd7e3c5f21984839b15179ed65b5098b52d6a758.jpg)

We list its 4 primary components - 

1.  **WebRTC stats:** Our VCA, Google Meet, uses webRTC, an open framework for the web that enables Real-Time Communications in the browser. The webRTC API allows capturing the mediaStream statistics including all the features we need. 
2.  **Throughput Prediction:** Uses an LSTM model to predict the throughput values (in packets/s) for the horizon size of 5 seconds
3.  **MPC Controller:** Includes 2 models for QoE estimate- frame jitter and frameRate, that utilize the predicted throughput. It solves the QoE optimization problem using a constraint solver.
4.  **Video Resolution Switching:** It receives the resolution values that optimize the QoE. And changes the resolution of the ongoing video call.

## Throughput Predictor Accuracy

RMSE of ?? was achieved using the curated dataset that is provided in this repo.

## Compared with an existing ABR

|   | QoE% increase |
| --- | --- |
| Google Meet | ?? |
| FaceTime | ?? |
| Zoom | ?? |