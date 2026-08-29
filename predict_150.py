from ultralytics import YOLO

# 1. 加载你刚从 Kaggle 偷运回来的核心科技
model = YOLO('best_150.pt')

# 2. 让它预测你的本地图片，save=True 表示自动把画好框的图片存下来
results = model('test_corn.jpg', save=True, conf=0.05, iou=0.3)

print("🎉 识别成功！快去项目里的 runs/detect/predict 文件夹里看画框效果图！")