# 202311-CobaPlus (A modified version of coba)


## Usage

### Execution
To conduct experiments, please execute the following: 
```shell script
python main.py <dataset name> <model name>
usage: main.py [-h] [-e EXPERIMENT] [-m MODEL_NAME] [-d DATA_DIR] [-me MAX_EPOCH] [-bs BATCH_SIZE] [--lr LR] [--device DEVICE] [-nw NUM_WORKERS] [-us] [-o OPTIMIZER] [-wd WEIGHT_DECAY]

```


- To send CSV by Gmail, you need to make or edit utils/gmail/account.json  
  ```json
  {
    "service_account": "GOOGLE_ACCOUNT",
    "password": "PASSWORD"
  }
  ```
  
  and utils/gmail/client_secret.json 
  - https://developers.google.com/gmail/api/  
  - https://console.cloud.google.com/
