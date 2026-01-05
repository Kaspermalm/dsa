.PHONY: backfill daily train infer clean all

all: clean backfill train infer

backfill:
	python src/backfill.py

daily:
	python src/daily.py

train:
	python src/train.py

infer:
	python src/inference.py

clean:
	rm -rf csv img model