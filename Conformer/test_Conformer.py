import csv
import predict_ten_steps
from predict_ten_steps import REPM
start_in=3000
test_repm=REPM(start_index=start_in)
with open("rmse.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["conformer"])
for i in range(1):
    next_ten_step_renewable_max, rmse =test_repm.pre(t=start_in+i)
    with open("rmse.csv", "a+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([rmse])
