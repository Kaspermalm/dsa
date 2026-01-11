.PHONY: backfill daily train infer clean all features train-hw infer-hw dashboard hindcast

all: clean backfill train infer hindcast

backfill:
	python src/backfill.py

daily:
	python src/daily.py

train:
	python src/train.py

infer:
	python src/inference.py

hindcast:
	python src/hindcast.py

features:
	python src/feature_pipeline.py

train-hw:
	python src/training_pipeline.py

infer-hw:
	python src/inference_pipeline.py

dashboard:
	streamlit run src/dashboard.py

clean:
	rm -rf csv img model
