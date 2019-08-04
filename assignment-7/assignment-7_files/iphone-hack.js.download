var md = new MobileDetect(window.navigator.userAgent);

/* yes, what follows is a disgusting hack -- I'm properly chagrined.

   If you read this and know how to
   disambiguate an iPhone 6 from a 12" iPad Pro only using CSS media
   queries (preferably using the bootstrap breakpoints), then send me
   an email at cscheid@cscheid.net and I'll buy you a drink of your
   choice. */

if (md.phone() == 'iPhone') {
    d3.select("head").append("link")
        .attr("rel", "stylesheet")
        .attr("href", "/css/iphone.css")
        .attr("type", "text/css");
}
