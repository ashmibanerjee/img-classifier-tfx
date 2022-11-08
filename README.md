# img-classifier-tfx

## Steps to run the model server
* `docker run --name myTFServing -it -v /Users/ashmi/Scripts/open-source/img-classifier-tfx:/img-classifier-tfx -p 9001:9001 --entrypoint /bin/bash tensorflow/serving
`
* `cd img-classifier-tfx/`
### REST API
* `tensorflow_model_server --rest_api_port=9001 --model_name=1662727000 --model_base_path=/img-classifier-tfx/src/models`
* `curl -X GET http:/localhost:9001/v1/models/1661517839`
* `curl -X POST http:/localhost:9001/v1/models/1661517839:predict -d @data.json -H "Content-Type: application/json"
`
### gRPC Server
* `tensorflow_model_server --port=9000 --model_name=1662731826 --model_base_path=/img-classifier-tfx/src/models`

### Running both `REST API` and `gRPC` together
* `docker run --name myTFServing -it -v /Users/ashmi/Scripts/open-source/img-classifier-tfx:/img-classifier-tfx -p 9001:9001 -p 9000:9000 --entrypoint /bin/bash tensorflow/serving`
* `tensorflow_model_server --port=9000 --rest_api_port=9001 --model_name=1662731826 --model_base_path=/img-classifier-tfx/src/models`

## References & Further Readings
* [https://towardsdatascience.com/deploying-machine-learning-models-with-tensorflow-serving-an-introduction-6d49697a1315](https://towardsdatascience.com/deploying-machine-learning-models-with-tensorflow-serving-an-introduction-6d49697a1315)
* [https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker](https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker)
* [https://towardsdatascience.com/image-classification-on-tensorflow-serving-with-grpc-or-rest-call-for-inference-fd3216ebd4f3](https://towardsdatascience.com/image-classification-on-tensorflow-serving-with-grpc-or-rest-call-for-inference-fd3216ebd4f3)
* [https://www.tensorflow.org/tfx/tutorials/serving/rest_simple](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
* [https://levelup.gitconnected.com/serving-an-image-classification-model-with-tensorflow-serving-c4657584d73d](https://levelup.gitconnected.com/serving-an-image-classification-model-with-tensorflow-serving-c4657584d73d)
* [https://towardsdatascience.com/image-classification-on-tensorflow-serving-with-grpc-or-rest-call-for-inference-fd3216ebd4f3](https://towardsdatascience.com/image-classification-on-tensorflow-serving-with-grpc-or-rest-call-for-inference-fd3216ebd4f3)
* [https://conferences.oreilly.com/tensorflow/tf-ca-2019/cdn.oreillystatic.com/en/assets/1/event/308/Advanced%20model%20deployments%20with%20TensorFlow%20Serving%20Presentation.pdf](https://conferences.oreilly.com/tensorflow/tf-ca-2019/cdn.oreillystatic.com/en/assets/1/event/308/Advanced%20model%20deployments%20with%20TensorFlow%20Serving%20Presentation.pdf)