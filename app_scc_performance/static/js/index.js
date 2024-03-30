$(document).ready(function(){
    
    // function to build tblMetrics
    function update_table(response){
        if(response.success == true){
            // get selected value in combobox
            var selected_model = $('#typeModel option:selected').text();

            // update table of metrics
            $('#tblMetrics tbody tr').empty().append(
                '<td>' + response.dataset + '</td>' +
                '<td>' + selected_model + '</td>' +
                '<td>' + response.rmse.toFixed(2) + '</td>' + 
                '<td>' + response.r2.toFixed(4) + '</td>'
            );
            // store download path
            $('#resultPath').val(response.download_path);
            console.log(response.download_path);

        }else{
            console.log('table cant load data');
        }
    }

    // function to build shap plot
    function load_shap(response){
        var shap_path = response.shap_path
        $('#shapImage').append('<img src=">' + shap_path + '" alt="Shap Bar plot">')
    }

    // generation of metric predictions
    $('form').submit(function(event){

        event.preventDefault();
        var formData = new FormData($(this)[0]);

        $.ajax({
            url: '',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response){
                // update table of metrics
                update_table(response);
                // load shap plot
                //load_shap(response);
            },
            error: function(xhr){
                console.log(xhr.status + ": " + xhr.responseText);
            }
        });
    });
    
});
