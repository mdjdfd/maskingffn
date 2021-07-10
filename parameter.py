import os
import uuid
import shutil
from datetime import datetime
import wticket as wt


def store_param():
    path_base = "storage"

    if not os.path.exists(path_base):
        os.makedirs(path_base)

    shutil.copy(os.path.basename(__file__), os.path.join(path_base, "wticket.py"))

    # for exp_id in range(10):
    for exp_id in range(1):
        date_now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        cur_exp = date_now + "-" + str(uuid.uuid4())
        path_cur_base = os.path.join(path_base, cur_exp)

        os.makedirs(path_cur_base)
        wt.run_model(path_cur_base)
