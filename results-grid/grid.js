agGrid.LicenseManager.setLicenseKey("YOUR_LICENSE_KEY_HERE");

document.addEventListener('DOMContentLoaded', function () {
    const gridOptions = {
        columnDefs: [
            {
                headerName: "", children: [
                    {field: 'TEST ID', rowDrag: true, filter: 'agNumberColumnFilter', checkboxSelection: true,},
                ]
            },
            {
                headerName: "TEST INSTANCES", children: [
                    {headerName: "FILE NAME", field: "FILE NAME", rowGroup: true,},
                    {headerName: "VEHICLE COUNT", field: "VEHICLE COUNT", filter: 'agNumberColumnFilter',},
                    {headerName: "TROLLEY COUNT", field: "TROLLEY COUNT", filter: 'agNumberColumnFilter',},
                    {
                        headerName: "TROLLEY IMPACT TIME",
                        field: "TROLLEY IMPACT TIME",
                        sortable: true,
                        filter: 'agNumberColumnFilter',

                    },
                    {
                        headerName: "EARLINESS/TARDINESS PENALTY",
                        field: "EARLINESS/TARDINESS PENALTY",
                        sortable: true,
                        filter: 'agNumberColumnFilter',
                    },
                ]
            },
            {
                headerName: "MATHEMATICAL FORMULATION", children: [
                    {
                        headerName: "Math Form Is Optimal OR Feasible",
                        field: "Math Form Is Optimal OR Feasible",
                        sortable: true,
                        filter: true,

                    },
                    {headerName: "Math CPU TIME", field: "Math CPU TIME", filter: 'agNumberColumnFilter',},
                    {headerName: "Math RESULT", field: "Math RESULT", filter: 'agNumberColumnFilter',}
                ]
            },
            {
                headerName: "ALNS", children: [
                    {headerName: "ALNS CPU TIME", field: "ALNS CPU TIME", filter: 'agNumberColumnFilter',},
                    {headerName: "ALNS RESULT", field: "ALNS RESULT", filter: 'agNumberColumnFilter',}
                ]
            },
            {
                headerName: "", children: [
                    {headerName: "GAP", field: "GAP", filter: 'agNumberColumnFilter',}
                ]
            },


        ], defaultColDef: {
            sortable: true,
            floatingFilter: true,
            pivot: true,
            enableRowGroup: true,
            enableValue: true,
            enablePivot: true,
            flex: 1,
        },
        groupDisplayType: "multipleColumns",
        rowDragManaged: true,
        rowSelection: "multiple",
        autoGroupColumnDef: {
            filter: "agGroupColumnFilter",
        },
        enableAdvancedFilter: true,
        rowGroupPanelShow: "always",
        enableRangeSelection: true,
        sideBar: ["columns", "filters"],
        statusBar: {
            statusPanels: [
                {statusPanel: "agTotalAndFilteredRowCountComponent"},
                {statusPanel: "agTotalRowCountComponent"},
                {statusPanel: "agFilteredRowCountComponent"},
                {statusPanel: "agSelectedRowCountComponent"},
                {statusPanel: "agAggregationComponent"},
            ],
        },
         enableCharts: true,
        rowData: rowData
    };

    const eGridDiv = document.querySelector('#myGrid');
    new agGrid.Grid(eGridDiv, gridOptions);
});