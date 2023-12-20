FROM jiaweitang/sbks-torch:latest
#FROM scratch

USER ${user}

#RUN pip freeze > ./requirements.txt

#RUN sed 's/==/>=/g' ./requirements.txt > ./TMP_FILE && \
#    mv ./TMP_FILE ./requirements.txt && \
#    pip install -r --user ./requirements.txt && \
#    rm -f ./requirements.txt

RUN pip install --user -U transformers && \
    pip install --user tokenizers==0.13.3 && \
    pip install --user accelerate && \
    pip install --user -U torch