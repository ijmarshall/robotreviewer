/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';

  var _ = require("underscore");

  // optimised CSS custom property getter/setter
  var CustomStyle = (function CustomStyleClosure() {

    // As noted on: http://www.zachstronaut.com/posts/2009/02/17/
    //              animate-css-transforms-firefox-webkit.html
    // in some versions of IE9 it is critical that ms appear in this list
    // before Moz
    var prefixes = ['ms', 'Moz', 'Webkit', 'O'];
    var _cache = {};

    function CustomStyle() {}

    CustomStyle.getProp = function get(propName, element) {
      // check cache only when no element is given
      if (arguments.length === 1 && typeof _cache[propName] === 'string') {
        return _cache[propName];
      }

      element = element || document.documentElement;
      var style = element.style, prefixed, uPropName;

      // test standard property first
      if (typeof style[propName] === 'string') {
        return (_cache[propName] = propName);
      }

      // capitalize
      uPropName = propName.charAt(0).toUpperCase() + propName.slice(1);

      // test vendor specific properties
      for (var i = 0, l = prefixes.length; i < l; i++) {
        prefixed = prefixes[i] + uPropName;
        if (typeof style[prefixed] === 'string') {
          return (_cache[propName] = prefixed);
        }
      }

      //if all fails then set to undefined
      return (_cache[propName] = 'undefined');
    };

    CustomStyle.setProp = function set(propName, element, str) {
      var prop = this.getProp(propName);
      if (prop !== 'undefined') {
        element.style[prop] = str;
      }
    };

    return CustomStyle;
  })();



  var TextLayerBuilder = function textLayerBuilder(options) {
    var viewport = options.viewport;
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');

    var calculateWidth = function(ctx, geom) {
      return ctx.measureText(geom.str).width;
    };

    this.isWhitespace = function(geom, style) {
      return (!/\S/.test(geom.str));
    };

    this.calculateStyles = function(geom, style) {
      var tx = PDFJS.Util.transform(viewport.transform, geom.transform);
      var angle = Math.atan2(tx[1], tx[0]);
      if (style.vertical) {
        angle += Math.PI / 2;
      }
      var fontHeight = Math.sqrt((tx[2] * tx[2]) + (tx[3] * tx[3]));
      var fontAscent = fontHeight;
      if (style.ascent) {
        fontAscent = style.ascent * fontAscent;
      } else if (style.descent) {
        fontAscent = (1 + style.descent) * fontAscent;
      }

      var left;
      var top;
      if (angle === 0) {
        left = tx[4];
        top = tx[5] - fontAscent;
      } else {
        left = tx[4] + (fontAscent * Math.sin(angle));
        top = tx[5] - (fontAscent * Math.cos(angle));
      }


      return {
        _angle: angle,
        fontSize: fontHeight + "px",
        fontFamily: style.fontFamily,
        left: left + 'px',
        top: top + 'px'
      };
    };

    this.createElement = function(geom, styles) {
      var style = this.calculateStyles(geom, styles[geom.fontName]);
      ctx.font = style.fontSize + ' ' + style.fontFamily;

      if(this.isWhitespace(geom, style)) {
        return {isWhitespace : true};
      }

      var width = calculateWidth(ctx, geom);
      if(width == 0) {
        return;
      }

      var textElement = {
        fontName:  geom.fontName,
        angle:  style._angle * (180 / Math.PI),
        style: style,
        textContent: geom.str
      };

      if (style.vertical) {
        textElement.canvasWidth = geom.height * viewport.scale;
      } else {
        textElement.canvasWidth = geom.width * viewport.scale;
      }

      var textScale = textElement.canvasWidth / width;
      var rotation = textElement.angle;
      var transform = 'scaleX(' + textScale + ')';
      transform = 'rotate(' + rotation + 'deg) ' + transform;

      CustomStyle.setProp('transform', textElement, transform);

      return textElement;
    };

    this.projectAnnotations = function(textElement, annotations) {
      if(!annotations || textElement.isWhitespace) {
        textElement.spans = null;
        textElement.annotations = null;
      } else {
        var color = annotations[0].color;
        textElement.color = color;

        var sorted = _.sortBy(annotations, function(ann) {// sorted by range offset
          return ann.range.lower;
        });

        var spans = sorted.map(function(ann, i) {
          var previous = sorted[i - 1];

          if(previous && previous.range.lower >= ann.range.lower && previous.range.upper >= ann.range.lower) {
            return null;
          }
          var next = sorted[i + 1];

          var text = textElement.textContent;
          if(!text) return null;

          var start = previous ? text.length + (previous.range.upper - previous.interval.upper) : 0,
              left = ann.range.lower - ann.interval.lower,
              right = text.length + (ann.range.upper - ann.interval.upper),
              end = next ?  right : text.length,
              style = {
                "opacity": 0.5,
                "backgroundColor": "rgb(" + ann.color.join(",") + ")"
              };

          return {
            pre: text.slice(start, left),
            content:text.slice(left, right),
            post: text.slice(right, end),
            style: style,
            color: color,
            uuid: ann.uuid
          };
        });
        textElement.spans = spans;
      }
    };

    this.createAnnotatedElement = function(geom, styles, ann) {
      var textElement = this.createElement(geom, styles);
      if(!textElement) {
        return {isWhitespace: true};
      }
      this.projectAnnotations(textElement, ann);
      return textElement;
    };
  };

  return TextLayerBuilder;
});
