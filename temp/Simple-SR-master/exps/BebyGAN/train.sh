# 실행할 Python 스크립트 파일 이름
PYTHON_SCRIPT="train.py"

# 출력을 저장할 파일 이름
OUTPUT_FILE="output.log"

# Python 스크립트를 실행하고 출력을 파일로 저장
python $PYTHON_SCRIPT > $OUTPUT_FILE 2>&1