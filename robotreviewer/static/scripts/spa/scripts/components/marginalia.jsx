/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2 -*- */
define(function (require) {
  'use strict';

  var _ = require("underscore");
  var $ = require("jquery");

  var React = require("react");

  var Editable = require("jsx!./editable");
  var Marked = require("marked");

  var Annotation = React.createClass({
    highlight: function(uuid) {
      $(window).trigger("highlight", this.props.annotation.get("uuid"));
    },
    destroy: function() {
      this.props.annotation.destroy();
    },
    select: function(annotation) {
      this.props.annotation.select();
    },
    render: function() {
      var annotation = this.props.annotation;
      var text = annotation.get("content");

      var isEditable = this.props.isEditable;
      var content = <a className="wrap" onClick={this.select}>{text}</a>;

      var remove = <i className="fa fa-remove remove" />;

      return (<li onMouseEnter={this.highlight} onMouseLeave={this.highlight}>
               {content} {isEditable ? <a onClick={this.destroy}>{remove}</a> : null}
             </li>);
    }
  });

  var Marginalis = React.createClass({
    toggleActivate: function(e) {
      var marginalia = this.props.marginalia;
      marginalia.toggleActive(this.props.marginalis);
    },
    setDescription: function(val) {
      this.props.marginalis.set("description", val);
    },
    render: function() {
      var marginalis = this.props.marginalis;
      var isEditable = this.props.isEditable;

      var description = marginalis.get("description");
      var isActive = marginalis.get("active");
      var style = {
        "backgroundColor": isActive ? "rgb(" + marginalis.get("color") + ")" : "inherit",
        "color": isActive ? "white" : "inherit"
      };



      var annotations = marginalis.get("annotations").map(function(annotation, idx) {
        return <Annotation annotation={annotation} isActive={isActive} isEditable={isEditable} key={idx} />;
      });

      var nAnnotations = <span className="annotations">{annotations.length}</span>;
      var icon = isActive ? <i className="fa fa-eye-slash"></i> :  <i className="fa fa-eye"></i>;
      var right = <span className="right">{icon}{nAnnotations}</span>;


      var content;
      if(isEditable) {
        content = <Editable content={description} callback={this.setDescription} />;
      } else {
        content = <div dangerouslySetInnerHTML={{__html: Marked(description)}}></div>;
      }

      return (<div className="block">
               <h4>
                 <a onClick={this.toggleActivate} style={style}>{marginalis.get("title")}{right}</a>
               </h4>
               <div className="content" style={{display: isActive ? "block" : "none"}}>
                 {content}
                 <ul className="no-bullet annotations">{annotations}</ul>
               </div>
              </div>);
    }
  });

  var Marginalia = React.createClass({
    getInitialState: function() {
      return { loading: false };
    },
    render: function() {
      if (this.state.loading) {
        var loader = require.toUrl(".") + "/../../img/loading-spin.svg";
        return <div className="loading"><img src={loader} viewBox="0 0 24 24" width="24" height="24" /></div>;
      }

      var isEditable = this.props.isEditable;
      var marginalia = this.props.marginalia;
      var grouped = marginalia.groupBy(function(m) { return m.get("type"); });

      var groups = _.map(grouped, function(group, type) {
        var blocks = group.map(function(marginalis, idx) {
          return <Marginalis key={idx} marginalia={marginalia} marginalis={marginalis} isEditable={isEditable} />;
        });
        return <div key={type} className="group"><h6 className="subheader">{type}</h6>{blocks}</div>;
      });

      return <div>{groups}</div>;
    }
  });

  return Marginalia;
});
