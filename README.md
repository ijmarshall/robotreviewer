# RobotReviewer
Automatic extraction of data from clinical trial reports

A simple webserver written in Python which accepts a clinical trial (as a PDF), and returns risk of bias judgements.

## Systematic review author?

This software is the *web-service* version, meaning it's aimed at people who make systematic review software.

**For most systematic review authors, if you want to try out RobotReviewer, you'd probably be better using the demo version on our website, available [here](https://robot-reviewer.vortext.systems).** If you like it, you could email the person who maintains your systematic review software a link to this site - they might be interested in adding it.

(Alternatively, individual authors who are adept at installing unix software from the terminal are free to install this version on their own machines by following the optional 'Web UI' instructions below).

## Developers of systematic review software?

RobotReviewer is open source and free to use under the GPL licence, version 3.0 (see the LICENCE.txt file in this repository).

We offer RobotReviewer free of charge, but we'd be most grateful if you would cite us if you use it. We're academics, and thrive on links and citations! Getting RobotReviewer widely used and cited helps us obtain the funding to maintain the project and make RobotReviewer better.

It also makes your methods transparent to your readers, and not least we'd love to see where RobotReviewer is used! :)

We'd appreciate it you could:

1. Display the text, 'Risk of Bias automation by RobotReviewer ([how to cite](http://vortext.systems/robotreviewer))' on the same screen or webpage on which the RobotReviewer results (highlighted text or risk of bias judgements) are displayed.
2. For web-based tools, the text 'how to cite' should link to our website `http://vortext.systems/robotreviewer`
3. For desktop software, you should usually link to the same website. If this is not possible, you may alternately display the text and example citations from the 'How to cite RobotReviewer' section below.

You can cite RobotReviewer as:

Marshall IJ, Kuiper J, & Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. Journal of the American Medical Informatics Association 2015. [doi:10.1093/jamia/ocv044](http://dx.doi.org/10.1093/jamia/ocv044)

A BibTeX entry for LaTeX users is

    @article{RobotReviewer2015,
      title = {{RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials}},
      author = {Marshall, Iain J and Kuiper, Jo\"{e}l and Wallace, Byron C},
      doi = {10.1093/jamia/ocv044},
      url = {http://dx.doi.org/10.1093/jamia/ocv044},
      journal = {Journal of the American Medical Informatics Association},
      year = {2015}
      month = jun,
      pages = {ocv044}
    }


## Installation

1. Ensure you have a working version of Python 2.7. We recommend using Python from the [Anaconda Python distribution](https://www.continuum.io/downloads) for a quicker and more reliable experience. However, if you have Python 2.7 already installed that will probably work fine too. (It's extremely close but won't yet work in Python 3 due to some weird unicode thing... don't laugh...)

2. Get a copy of the RobotReviewer repo, and go into that directory
    ```bash
    git clone https://github.com/ijmarshall/robotreviewer3.git
    cd robotreviewer3
    ```

3. Get the large data files (>1GB) via git lfs (`brew install git-lfs` if you don't have it)
    ```bash
    git lfs install
    git lfs pull
    ```
<!-- 3. ignore point 3 for now...  -->
<!-- 3. Install the PDF web viewer (optional --- this is not needed if you want to just use the REST API)
    ```bash
    git submodule update --init --recursive
    ```
 -->
4. Install the Python libraries that RobotReviewer needs - do one of the following.

    a. If you are using Anaconda:

```bash
conda config --add channels spacy  # only needed once
conda install flask numpy scipy scikit-learn spacy
pip install fuzzywuzzy # (this is not yet in the anaconda repo)
```

    b. For everyone else:

```bash
pip install flask numpy scipy scikit-learn spacy fuzzywuzzy
```

5. Install the sentence processing data:
    ```bash
    python -m spacy.en.download
    ```
      
6. This version of RobotReviewer requires Grobid, which runs on Java. Follow the instructions [here](https://grobid.readthedocs.io/en/latest/Install-Grobid/) to download and build it.

7. Edit the `robotreviewer/config.py` file to contain the path to the directory where you have installed Grobid. (RobotReviewer will start it automatically in a subprocess).

## Running

The following

```bash
python -m robotreviewer
```

will start a flask server running on `http://localhost:5000`. You can run the server in development mode by passing `DEBUG=true python -m robotreviewer`.

## REST API

1. To annotate plain text:
    Send some JSON by POST to `/annotate` such as:
    ```json
    {"text": "Put the full text of a clinical trial extracted from the PDF in here"}
    ```

    and it will return something like:

    ```json
    {"marginalia": [
       {"title":"Random sequence generation",
        "type":"Risk of Bias",
        "description":"**Overall risk of bias prediction**: low",
        "annotations":[
           {"content":"A central pharmacy randomly assigned study medication in a 1:1 ratio using a computer-generated randomization sequence with variable-sized blocks ranging from 2 to 8 stratified by study site.",
            "uuid":"6e97f8d0-2970-11e5-b5fe-0242ac110006"
           }, ...
    ```

2. To annotate a PDF:
    Send a PDF by to: `/annotate_pdf`
    (Notes for Joël: 1. this works with my slightly edited version of Spa... I've pushed it not as a submodule for the moment since I don't want to mess with the proper Spa. 2. I had some trouble getting the file data from Spa using the conventional Flask method. I *believe* it's since Spa uploads the PDF as data, and not as a multi-part encoded file. It works now, but just so you know it's a bit different from the report view thing below.)

3. Report view!:
    Send a zip file containing RCT PDFs to: `/generate_report`

    **N.B. [here](https://gist.github.com/ijmarshall/2298e68f780e7287a14fb97f97f62085#file-report_view_demo-ipynb) is a working notebook which demonstrates this function**


## Help

Feel free to contact us on [mail@ijmarshall.com](mailto:mail@ijmarshall) with any questions.

## References

1. Marshall, I. J., Kuiper, J., & Wallace, B. C. (2015). RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. Journal of the American Medical Informatics Association. [[doi]](http://dx.doi.org/10.1093/jamia/ocv044)
2. Marshall, I., Kuiper, J., & Wallace, B. (2015). Automating Risk of Bias Assessment for Clinical Trials. IEEE Journal of Biomedical and Health Informatics. [[doi]](http://dx.doi.org/10.1109/JBHI.2015.2431314)
3. Kuiper, J., Marshall, I. J., Wallace, B. C., & Swertz, M. A. (2014). Spá: A Web-Based Viewer for Text Mining in Evidence Based Medicine. In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD 2014) (Vol. 8726, pp. 452–455). Springer Berlin Heidelberg. [[doi]](http://dx.doi.org/10.1007/978-3-662-44845-8_33)
4. Marshall, I. J., Kuiper, J., & Wallace, B. C. (2014). Automating Risk of Bias Assessment for Clinical Trials. In Proceedings of the ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB) (pp. 88–95). ACM. [[doi]](http://dx.doi.org/10.1145/2649387.2649406)

Copyright (c) 2015 Iain Marshall, Joël Kuiper, and Byron Wallace
