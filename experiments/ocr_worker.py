"""JSON-lines OCR worker; deliberately independent of Torch and the training env."""
import argparse
import base64
from contextlib import redirect_stdout
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='PP-OCRv5_server_rec')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    args = p.parse_args()
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    with redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        from paddleocr import TextRecognition
        model = TextRecognition(model_name=args.model, device=args.device,
                                enable_mkldnn=False, cpu_threads=4)
    print(json.dumps(dict(ready=True)), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            images = [cv2.imdecode(np.frombuffer(base64.b64decode(x), np.uint8), cv2.IMREAD_COLOR)
                      for x in request['images']]
            if any(x is None for x in images):
                raise ValueError('Invalid crop PNG')
            with redirect_stdout(sys.stderr):
                results = [(x['rec_text'], float(x['rec_score'])) for x in
                           model.predict(images, batch_size=args.batch_size)]
            if len(results) != len(images):
                raise ValueError('Recognition count mismatch')
            print(json.dumps(dict(predictions=results), ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps(dict(error=f'{type(e).__name__}: {e}')), flush=True)


if __name__ == '__main__':
    main()
