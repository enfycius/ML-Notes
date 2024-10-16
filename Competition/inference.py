import datetime

class Model:
    """
    각 팀에서 사용하는 모델에 대한 정의
    혹은 직접 Model이 정의된 파일을 업로드하고 import 무방
    """



def get_external_env_data(window_size, target_datetime):
    """
    Args:
        window_size: prediction을 위해 추출할 외부환경 데이터의 n수
        target_datetime: prediction 대상 일시

    Returns:
        list(multi-dimensional): 추론에 사용할 외부환경 시계열 데이터셋

    Note:
        get_input_dataset에서 활용할 수 있도록 작성(불필요시 작성X)
    """

def get_control_data(window_size, target_datetime):
    """
    Args:
        window_size: prediction을 위해 추출할 제어 데이터의 n수
        target_datetime: prediction 대상 일시

    Returns:
        list(multi-dimensional): 추론에 사용할 제어 시계열 데이터셋

    Note:
        get_input_dataset에서 활용할 수 있도록 작성(불필요시 작성X)
    """

def get_initial_env_data(target_datetime):
    """
    Args:
        target_datetime: prediction 대상 일시

    Returns:
        list: 추론에 사용할 초기 내부환경 데이터셋

    Note:
        get_input_dataset에서 활용할 수 있도록 작성(불필요시 작성X)

    """

def get_input_dataset(target_datetime):
    """
    Args:
        target_datetime: prediction 대상 일시


    Returns:
        list(multi-dimensional): 추론에 사용할 input dataset

    Note:
        팀별로 정의한 모델의 input이 되는 데이터를 생성하여 리턴하는 함수
        Return하는 list는 predict 함수의 input_data Parameter로 사용
    """

def load_model(model_path):
    """
    Args:
        model_path: 모델 파일이 있는 경로

    Returns:
        Object: 로드한 모델 객체

    Note:
        팀별로 사전에 학습한 pth와 같은 weight 파일을 로드하는 함수
        Return하는 Model Object는 predict 함수의 model Parameter로 사용
    """

def predict(model, input_data):
    """
    Args:
        model: 팀별로 구축한 모델을 로드한 파일
        input_data: 특정 시점의 내부 환경데이터 예측을 위한 앞선 기간의 외부환경데이터/제어데이터/초기내부환경데이터 (팀별 모델에 따라 각기 다름)

    Returns:
        tuple: 특정 시점의 내부 환경데이터 예측 값 (CO2, 습도, 온도) 순서로 반드시 1x3 크기의 tuple로 return
    """



### 테스트용 코드
### 반드시 테스트 코드가 정상 동작하여 출력이 되는지 확인 후 제출하세요
def test():
    model_path = './model.pth'
    target_datetime = datetime.datetime.strptime("2024-08-08 00:00", "%Y-%m-%d %H:%M")
    y_pred = predict(load_model(model_path), get_input_dataset(target_datetime))
    print(y_pred)


test()