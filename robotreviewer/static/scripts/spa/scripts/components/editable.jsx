/* -*- mode: js2; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js2-basic-offset: 2 -*- */

define(function (require) {
  'use strict';

  var Marked = require("marked");
  var React = require("react");

  var Editable = React.createClass({
    getInitialState: function() {
      return { editable: false };
    },
    edit: function() {
      this.setState({ editable: true});
    },
    submit: function(e) {
      this.setState({ editable: false });
      this.props.callback(this.refs.input.getDOMNode().value);
      e.preventDefault();
    },
    componentDidUpdate: function() {
      if(this.state.editable) {
        this.refs.input.getDOMNode().focus();
      }
    },
    render: function() {
      var content = this.props.content || "*Click to edit*";
      if(this.state.editable) {
        return (
            <form className="row collapse">
              <div className="small-10 columns">
                <input type="text" ref="input" defaultValue={this.props.content}></input>
              </div>
              <div className="small-2 columns">
                <button href="#" className="button postfix" onClick={this.submit}>Edit</button>
              </div>
            </form>
        );
      } else {
        return (
            <div className="editable" onClick={this.edit}>
              <div dangerouslySetInnerHTML={{__html: Marked(content)}}></div>
            </div>);
      }
    }
  });

  return Editable;

});
