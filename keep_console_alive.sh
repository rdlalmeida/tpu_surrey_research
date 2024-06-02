#!/usr/local/nvm/versions/node/v20.13.0/bin/node
// Essentially, this creates a periodic function that, every 30000ms, i. e., 30s, puts the cursor at the position x=500, y=500, and "clicks" it, which shouldn't do a thing other than resetting the annoying disconnecting interval from Google Cloud Console
let refreshInterval = 300000;

setInterval(function() {
	document.elementFromPoint(500,500).click();	
}, refreshInterval);

console.log("Enabled screen refresh every %s seconds", (refreshInterval/1000));
