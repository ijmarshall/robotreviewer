/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';
  var React = require("react");

  var Dropzone = require("react-dropzone")
  var FileUtil = require("spa/helpers/fileUtil");
  var _ = require("underscore");
  var Q = require("Q");
  var $ = require("jquery");

  var uploadUri = "/add_pdfs_to_db";
  var synthesizeUri = "/synthesize_uploaded";

  var ReportView = React.createClass({
    onDrop: function (files) {
      var fd = new FormData();

      _.forEach(files, function(file) {
        fd.append('file', file);
      });

      $.ajax({
        type: 'POST',
        url: uploadUri,
        data: fd,
        contentType: false,
        cache: false,
        processData: false,
        success: function(data) {
          $.post(synthesizeUri, {}, function(data, status) {
            console.log(data, status);
          });
        }
      });
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
