import csv
import numpy as np
import matplotlib.pyplot as plt


plot_num = 2

if plot_num == 1 :
    csv_path = '/Users/nak/Library/CloudStorage/GoogleDrive-katsuyuki.nakamura.cv@gmail.com/マイドライブ/python_image_recognition/6_img_captioning/my_script/show_and_tell.csv'
    out_pdf = '/Users/nak/Library/CloudStorage/GoogleDrive-katsuyuki.nakamura.cv@gmail.com/マイドライブ/python_image_recognition/6_img_captioning/my_script/show_and_tell_learning_curve.pdf'
    title_str = 'Show and tell'
elif plot_num == 2 :
    csv_path = '/Users/nak/Library/CloudStorage/GoogleDrive-katsuyuki.nakamura.cv@gmail.com/マイドライブ/python_image_recognition/6_img_captioning/my_script/show_attend_and_tell.csv'
    out_pdf = '/Users/nak/Library/CloudStorage/GoogleDrive-katsuyuki.nakamura.cv@gmail.com/マイドライブ/python_image_recognition/6_img_captioning/my_script/show_attend_and_tell_learning_curve.pdf'
    title_str = 'Show, attend and tell'
elif plot_num == 3 :
    csv_path = '/Users/nak/Library/CloudStorage/GoogleDrive-katsuyuki.nakamura.cv@gmail.com/マイドライブ/python_image_recognition/6_img_captioning/my_script/transformer_captioning.csv'
    out_pdf = '/Users/nak/Library/CloudStorage/GoogleDrive-katsuyuki.nakamura.cv@gmail.com/マイドライブ/python_image_recognition/6_img_captioning/my_script/transformer_captioning_learning_curve.pdf'
    title_str = 'Transformer captioning'

rows = []
with open(csv_path) as f:   
    reader = csv.reader(f)
    rows = [row for row in reader]

header = rows.pop(0)

data = np.float_(np.array(rows).T)

plt.xlabel(header[0])
plt.ylabel(header[1])
plt.plot(data[0], data[1], linestyle='solid')
plt.plot(data[0], data[2], linestyle='solid')

plt.title(title_str)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend(["Training", "Validation"])
plt.axis([0, 100, 2.0, 5.5])
axes=plt.gca()
axes.set_aspect(20)


plt.savefig(out_pdf, dpi=300)

print('finish')
