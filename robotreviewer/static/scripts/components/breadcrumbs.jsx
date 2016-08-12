/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';
  var React = require("react");
  var _ = require("underscore");

  var Breadcrumbs = React.createClass({
    render: function() {
      var b = this.props.breadcrumbs;

      var breadcrumbs = b.map(function(crumb, index) {
        var link = crumb.get("link");
        var title = crumb.get("title");
        if(index === b.length - 1) {
          return(<span className="current" key={link}>{title}</span>);
        } else {
          return(<a href={link} key={link}>{title}</a>);
        }
      });

      return(<nav className="breadcrumbs">{breadcrumbs}</nav>);
    }
  });

  return Breadcrumbs;

});
