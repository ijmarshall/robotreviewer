/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';
  var React = require("react");

  var Dropzone = require("react-dropzone")
  var _ = require("underscore");
  var $ = require("jquery");

  var uploadUri = "/upload_and_annotate_pdfs";

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
                self.setState({message: "Thinking…", progress: ""});
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
          var resp = JSON.parse(data);
          var reportId = resp["report_uuid"];
          var pollDelay = 1000; // in ms
          var monitorProgress = function(){
            $.getJSON('annotate_status/' + reportId, function(status_data){
                if (status_data.state == 'SUCCESS') {
                    self.setState({inProgress: false, message: ""});
                    window.router.navigate('report/' + reportId, {trigger: true});
                } else {
                    self.setState({inProgress: true, message: status_data.meta.process_percentage + '% done: ' + status_data.meta.task});
                    setTimeout(monitorProgress, pollDelay);
                };
            });
          };
          setTimeout(monitorProgress, pollDelay);
        },
        error: function (xhr, ajaxOptions, thrownError) {
          // probably want to yell @ the user here?
          self.setState({inProgress: false, error: "We're sorry, something went wrong!"});
        }
      });
    },
    render: function() {
      var inProgress = this.state.inProgress;
      var error = this.state.error ? <div className="alert-box alert">{this.state.error}</div> : null;

      var stopPropagation = function(e) {
        if (!e)
          e = window.event;

        //IE9 & Other Browsers
        if (e.stopPropagation) {
          e.stopPropagation();
        }
        //IE8 and Lower
        else {
          e.cancelBubble = true;
        }
      };

      return (
          <div className="upload" style={{position: "relative", zIndex: 10, top: "-100px"}}>
          {error}
          <div style={{opacity: inProgress ? 1 : 0, zIndex: 100, position: "relative"}} className="infinity">
            <div>
              <img src="/img/infinity.gif" width="120" height="120" />
              <br />
              {this.state.message + " " + this.state.progress}
            </div>
          </div>
          <div style={{display: inProgress ? "none" : "block", "z-index": 10, "position": "relative"}}>
            <Dropzone onDrop={this.onDrop}
                      accept="application/pdf"
                      disablePreview={true}
                      activeClassName="dropzone-active"
                      className="dropzone">

                <div>
                RobotReviewer automatically extracts and synthesises data from Randomized Controlled Trials.
                <br />
                Drag and drop PDFs here, or click to select files to upload!

                <br />
                Or click one of the examples: <a href="/#report/Tvg0-pHV2QBsYpJxE2KW-" onClick={stopPropagation}>Decision aids</a>, <a href="/#report/_fzGUEvWAeRsqYSmNQbBq" onClick={stopPropagation}>Influenza vaccination</a>, <a href="/#report/HBkzX1I3Uz_kZEQYeqXJf" onClick={stopPropagation}>Hypertension</a>
                </div>

              </Dropzone>
            </div>
          </div>
      );
    }
  });

  return UploadView;
});
