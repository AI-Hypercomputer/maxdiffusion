import requests
import time
r = requests.get("http://127.0.0.1:8080/health")
print(r.status_code, r.reason)
iters = 10
for _ in range(iters):
  s = time.time()
  r = requests.post("http://127.0.0.1:8080/predict",
                    json={"instances": [
                            {
                              "prompt" : ["a dog walking a cat"],
                              "query_id" : ["1"]
                            },
                            {
                              "prompt" : ["a dog walking a cat"],
                              "query_id" : ["1"]
                            },
                            {
                              "prompt" : ["a dog walking a cat"],
                              "query_id" : ["1"]
                            }
                          ]
                          },
                          headers={"Content-Type": "application/json"},
                          )
  print(r.status_code, r.reason)
  print("request time: ", (time.time() - s))

  with open("response.json", "w") as f:
      f.write(r.text)
  print("total time: ", (time.time() - s))
