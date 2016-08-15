/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';
  var React = require("react");

  var Dropzone = require("react-dropzone")
  var _ = require("underscore");
  var $ = require("jquery");

  var uploadUri = "/upload_and_annotate";

  var UploadView = React.createClass({
    getInitialState: function() {
      return { progress: "", inProgress: false, message: "", error: ""};
    },
    onDrop: function (files) {
      var self = this;
      var fd = new FormData();



      _.forEach(files, function(file) {
        fd.append('file', file);
      });

      $.ajax({
        xhr: function() {
          self.setState({inProgress: true, message: "Uploading documents..."});
          var xhr = new window.XMLHttpRequest();
          xhr.upload.addEventListener("progress", function(evt) {
            if (evt.lengthComputable) {
              var percentComplete = evt.loaded / evt.total;
              percentComplete = parseInt(percentComplete * 100);
              if(percentComplete < 100) {
                self.setState({progress: percentComplete + "%"});
              } else {
                self.setState({message: "Synthesizing predictions...", progress: ""});
              }
            }
          }, false);
          return xhr;
        },
        type: 'POST',
        url: uploadUri,
        data: fd,
        contentType: false,
        cache: false,
        processData: false,
        success: function(data) {
          self.setState({inProgress: false, message: ""});
          var resp = JSON.parse(data);
          var reportId = resp["report_uuid"];
          window.router.navigate('report/' + reportId, {trigger: true});
        },
        error: function (xhr, ajaxOptions, thrownError) {
          // probably want to yell @ the user here?
          self.setState({inProgress: false, error: "We're sorry, something went wrong!"});
        }
      });
    },
    render: function() {
      var inProgress = this.state.inProgress;
      var progress = this.state.progress;
      var error = this.state.error ? <div className="alert-box alert">{this.state.error}</div> : null;
      return (
          <div className="upload">
          {error}
          <div style={{opacity: inProgress ? 1 : 0}} className="infinity">
            <div>
              <img src="/img/infinity.gif" width="120" height="120" />
              <br />
              {this.state.message + " " + progress}
            </div>
          </div>
          <div style={{display: inProgress ? "none" : "block"}}>
            <Dropzone onDrop={this.onDrop}
                      accept="application/pdf"
                      disablePreview={true}
                      activeClassName="dropzone-active"
                      className="dropzone">

                <div>
                RobotReviewer helps to automate systematic reviews in Evidence Based Medicine.
                <br />
                Try dropping Randomized Controlled Trial PDFs here, or click to select files to upload!
                </div>
              </Dropzone>
            </div>
          </div>
      );
    }
  });

  return UploadView;
});
