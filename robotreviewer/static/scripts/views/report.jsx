/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';
  var React = require("react");

  var Dropzone = require("react-dropzone");

  var ReportView = React.createClass({
    onDrop: function (files) {
      console.log('Received files: ', files);
    },
    render: function() {
      return (
          <div>
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
      );
    }
  });

  return ReportView;
});
