# Full Stack Deep Learning Labs

Welcome!

Project developed during lab sessions of the [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com).

- We will build a handwriting recognition system from scratch, and deploy it as a web service.
- Uses Keras, but designed to be modular, hackable, and scalable
- Provides code for training models in parallel and store evaluation in Weights & Biases
- We will set up continuous integration system for our codebase, which will check functionality of code and evaluate the model about to be deployed.
- We will package up the prediction system as a REST API, deployable as a Docker container.
- We will deploy the prediction system as a serverless function to Amazon Lambda.
- Lastly, we will set up monitoring that alerts us when the incoming data distribution changes.

Sequence:

- [Setup](setup/readme.md)
- [Lab 1](lab1/readme.md): Introduction to the codebase by training a simple character predictor.
- [Lab 2](lab2/readme.md): Introduce EMNIST, make a syntehtic dataset, and try to recognize text.
- [Lab 3](lab3/readme.md): LSTM + CTC based approach to line text recognition.
- [Lab 4](lab4/readme.md): Transformer-based approach to line text recognition.
- [Lab 5](lab5/readme.md): IAM Lines, Weights & Biases, and hyperparameter sweeps.
- [Lab 5](lab5/readme.md): Train and evaluate line detection model (or paragraph recognition).
- [Lab 7](lab6/readme.md): Label our own handwriting data, download and version results.
- [Lab 8](lab7/readme.md): Add continuous integration that runs linting and tests on our codebase.
- [Lab 9](lab8/readme.md): Run as a REST API first locally, then in a container, and then deploy to the web.
- [Lab 10](lab10/readme.md): Set up monitoring that alerts us when the incoming data distribution changes.
