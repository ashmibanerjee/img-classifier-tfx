# img-classifier-tfx

## Steps to run the model server
* `docker run --name myTFServing -it -v /Users/ashmi/Scripts/img-classifier-tfx:/img-classifier-tfx -p 9001:9001 --entrypoint /bin/bash tensorflow/serving
`
* `cd img-classifier-tfx/`
* `tensorflow_model_server --rest_api_port=9001 --model_name=1657646009 --model_base_path=/img-classifier-tfx/src/pred/models`
* `curl -X GET http:/localhost:9001/v1/models/1657646009`
* `curl -X POST http:/localhost:9001/v1/models/1657646009:predit -d @data.json -H "Content-Type: application/json"
`

## References & Further Readings
* [https://towardsdatascience.com/deploying-machine-learning-models-with-tensorflow-serving-an-introduction-6d49697a1315](https://towardsdatascience.com/deploying-machine-learning-models-with-tensorflow-serving-an-introduction-6d49697a1315)
* [https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker](https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker)
* [https://towardsdatascience.com/image-classification-on-tensorflow-serving-with-grpc-or-rest-call-for-inference-fd3216ebd4f3](https://towardsdatascience.com/image-classification-on-tensorflow-serving-with-grpc-or-rest-call-for-inference-fd3216ebd4f3)
* [https://www.tensorflow.org/tfx/tutorials/serving/rest_simple](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
* [https://levelup.gitconnected.com/serving-an-image-classification-model-with-tensorflow-serving-c4657584d73d](https://levelup.gitconnected.com/serving-an-image-classification-model-with-tensorflow-serving-c4657584d73d)