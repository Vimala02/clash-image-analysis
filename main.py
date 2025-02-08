pip install --upgrade torchfrom waitress import serve
from deploy import app
import multiprocessing

if __name__ == '__main__':
    num_cups = multiprocessing.cpu_count()
    serve(app, host='0.0.0.0', port=8000)
    print('serve ended')