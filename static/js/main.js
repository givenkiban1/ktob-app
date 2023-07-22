$("#statusMessage").hide();

function getLoadingAnimation() {

  var load = document.createElement('div');
  load.className = 'loader';
  return load
}


// Send the CSV file to the server using AJAX
$("#uploadBtn").on("click", function () {

  $("#loadStuff").html(getLoadingAnimation());

  var formData = new FormData($('#csvForm')[0]);
  $.ajax({
    url: "/upload",
    type: "POST",
    data: formData,
    contentType: false,
    processData: false,
    success: function (data) {
      if ('error' in data) {
        alert(data.error);
      } else {
        // Create an image element and display the generated plot image
        $("#formStuff").hide();
        // JavaScript with jQuery
        var imageList = data.images;
        var currentIndex = 0;

        var statusMessages = [
          "Plotting routes on map ...",
          "Detecting centroids ...",
          "Plotting boundary box ...",
          "Finding the shortest path ..."
        ];

        function showNextImage() {
          if (currentIndex >= imageList.length) {
            $("#statusMessage").text(statusMessages[currentIndex]);
            return;
          }

          var img = new Image();

          img.src = "data:image/png;base64," + imageList[currentIndex];
          $("#plotDiv").html(img);

          $("#statusMessage").text(statusMessages[currentIndex]);
          $("#statusMessage").show();

          // $('#imageContainer').html('<img src="' + imageUrl + '">');
          currentIndex++;

          // Set the interval to display the images with a delay of 2000ms (2 seconds)
          setTimeout(showNextImage, 3000);

        }

        // Start showing the images
        showNextImage();

        generateMap();

      }

    },
    error: function () {
        alert("An error occurred while uploading the CSV file.");
        $("#loadStuff").empty();
    }
  });
});

function generateMap(){


  var formData = new FormData($('#csvForm')[0]);
  $.ajax({
    url: "/generateMap",
    type: "POST",
    data: formData,
    contentType: false,
    processData: false,
    success: function (data) {

      // alert(data.success);
      $("#loadStuff").empty();

      $("#plotDiv").empty();

      $("#statusMessage").text("Completed! Download your PDFs below.");

      var iframe = document.createElement('iframe');
      iframe.src = "/iframe";
      iframe.width = 700;
      iframe.height = 400;
      $("#iframe").html(iframe);


      // download pdf button
      var a = document.createElement('a');
      a.textContent = "Download PDFs";
      a.className = "download-button";
      a.href="/download";
      a.download = "results.zip";

      // reset button
      var resetButton = $('<button>')
        .addClass('reset-button')
        .html('<i class="fas fa-undo"></i> Reset')
        .click(function () {
          // Reload the page to reset the application
          location.reload();
        });

      $("#download").html(a).after(resetButton);


    },
    error: function(){
      $("#loadStuff").empty();
      alert("An error occurred while receiving the map image.");
    }
  });


}
