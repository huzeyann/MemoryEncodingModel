FROM huzeeee/afo:latest

WORKDIR /workspace

# RUN pip install --upgrade open_clip_torch --no-dependencies
# RUN pip install --upgrade timm --no-dependencies
# RUN pip install git+https://github.com/facebookresearch/dinov2.git --no-dependencies --ignore-requires-python

RUN pip install fast-pytorch-kmeans --no-dependencies

CMD ["sleep", "infinity"]
