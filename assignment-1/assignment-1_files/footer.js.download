$(document).ready(function() {
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
});

function guessGithubURL() {
    var path = window.location.pathname;
    if (path.endsWith("/"))
        path += "index.md";
    else if (path.endsWith(".html"))
        path = path.substr(0, path.length-4) + "md";
    return "http://github.com/cscheid/cscheid.net/edit/master" + path;
}

d3.select("#pull-request")
    .attr("href", guessGithubURL());

function footnoteCumulativeHeights() {
    var heights = d3.select("#footnotes-ol-sidebar")
        .selectAll("li")
        .nodes()
        .map((node) => node.getBoundingClientRect().height);
    var totalSoFar = 0;
    return heights.map((h) => {
        var result = totalSoFar;
        totalSoFar += h;
        return result;
    });
}

// based on http://stackoverflow.com/a/1480137/221007
function cumulativeOffset(element, parent) {
    var top = 0, left = 0;
    do {
        top += element.offsetTop  || 0;
        left += element.offsetLeft || 0;
        element = element.offsetParent;
    } while(element && element !== parent);

    return {
        top: top,
        left: left
    };
};

function extractFootnotes() {
    var div = d3.select("div.footnotes");
    if (div.nodes().length > 0) {
        if (d3.select("#footnotes-ol-sidebar").nodes().length === 0) {
            // no sidebar, instead add a footnotes section
            d3.select(div.node().parentNode).insert("h2", "div.footnotes").text("Footnotes");
            return;
        }

        // first, build slightly-incorrect height-adjusted footnotes.
        div.select("ol").selectAll("li").each(function(d, index) {
            // can't d3.select() here because using the id actually makes an invalid
            // selector (it has a colon in it)
            var refId = d3.select(this).selectAll("a.reversefootnote").attr("href").substr(1);
            var link = document.getElementById(refId);
            var offset = cumulativeOffset(link, d3.select("#content").node());
            var that = this;
            var dthis = d3.select(this);
            dthis.remove();
            d3.select("#footnotes-ol-sidebar").append(function() { return that; });
            d3.select(this)
                .style("position", "relative")
                .style("top", offset.top + "px");
        });
        
        // now, correct the heights. 
        var cumulativeHeights = footnoteCumulativeHeights();
        d3.select("#footnotes-ol-sidebar")
            .selectAll("li")
            .each(function(d, index) {
                var top = Number(this.style.top.substr(0, this.style.top.length-2));
                this.style.top = (top - cumulativeHeights[index]) + "px";
            });

        // FIXME there's still a bug where the footnotes can overlap one another.
    }
}
extractFootnotes();
