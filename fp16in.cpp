#include "utility/common.hpp"




int main(int argc, char** argv){
    DeviceProp prop;
    getDeviceProperties(&prop, 0);
    std::string GPUModel=prop.name;


    FP16 *h_a, *h_b, *h16_c, *d16_a, *d16_b, *d16_c,
    one16=__float2half(1.), minsubnormal16 = __float2half(ldexp(1., -24)),
    minnormal16= __float2half(ldexp(1., -14)),
    belowone16 = __float2half(1. - ldexp(1, -11));
    FP32 *d_c, *h_c, belowone = nextafterf(1., 0.); 
    int index;

    FILE *outfile = stdout;
    uint32_t c_print, c_compare;

    h_a = new FP16[16*16];
    h_b = new FP16[16*16];
    h_c = new FP32[16*16];
    h16_c = new FP16[16*16];

    CHECK_DEVICE_ERROR(deviceMalloc((void**)&d16_a, 16*16*sizeof(FP16)));
    CHECK_DEVICE_ERROR(deviceMalloc((void**)&d16_b, 16*16*sizeof(FP16)));
    CHECK_DEVICE_ERROR(deviceMalloc((void**)&d16_c, 16*16*sizeof(FP16)));
    CHECK_DEVICE_ERROR(deviceMalloc((void**)&d_c, 16*16*sizeof(FP32)));

    /*
     ==========================  ========================== Test subnormals ==========================  ========================== 
    */
    printMainTest("Test subnormals' features.");
    /*
    Test subnormal in a and b
    */
    printSubTest("Test subnormals' inputs for A and B. ");
    bool support_sub_input = 0;
    host_reset(h_a, h_b, h_c);
    h_a[0] = minsubnormal16;
    h_b[0] = one16;
    h_c[0] = 0.;
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == ldexp(1., -24)){
        support_sub_input = 1;
        printText("Support subnormal inputs for A ");
    }
    else {
        memcpy(&c_print, &h_c[0], sizeof(FP32));
        std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
        printText("NOT support subnormal inputs for A ");
    }
    host_reset(h_a, h_b, h_c);
    h_a[0] = one16;
    h_b[0] = minsubnormal16;
    h_c[0] = 0.;
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == ldexp(1., -24)){
        support_sub_input = support_sub_input & 1;
        printText("Support subnormal inputs for B ");

    }
    else {
        support_sub_input = 0;
        memcpy(&c_print, &h_c[0], sizeof(FP32));
        std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
        printText("NOT support subnormal inputs for B ");
    }


    /*
    Test subnormals in c (and d, two fp16 cannot output fp32 subnormals)
    */
    printSubTest("Test subnormals' (outputs for D and) inputs for C.");
    printText("Any FP16's additions won't restult in FP32'a subnormals, so no need to test subnormals for D.");
    host_reset(h_a, h_b, h_c);
    h_a[0] = 0;
    h_b[0] = 0;
    h_c[0] = ldexp(1., -127);
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == ldexp(1., -127)){
        support_sub_input = support_sub_input & 1;
        printText("Support subnormal inputs for C. ");
    }
    else {
        support_sub_input = 0;
        memcpy(&c_print, &h_c[0], sizeof(FP32));
        std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
        printText("NOT support subnormal inputs for C. ");
    }

    // if(support_sub_input) {
    //     printitem(outfile, "Subnormal are supported in all inputs. ");
    // }

    // printfooter(outfile);


    /*
     ==========================  ========================== Test Rounding mode for inner product addition ==========================  ========================== 
    */

    printMainTest("Testing Rounding mode and Extra bits. ");

    printSubTest("Test initial rounding mode for inner product addition.(a11b11+a12b12)");

    /*
    Add postive numbers and rounding
    */

    bool rtz=0, ru=0, rd=0, rtn=0;

    host_reset(h_a, h_b, h_c);
    h_a[0] = minnormal16;
    h_a[1] = one16;
    h_b[0] = __float2half(ldexp(1., -9) + ldexp(1., -10));
    h_b[1] = __float2half(2.);
    wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == 2.){
        printText("inner product addition rounding down or rounding to zero.");
        rd = 1;
        rtz = 1;
    }
    else if(h_c[0] == 2. + ldexp(1., -22)){
        printText("inner product addition rounding up or rounding to nearest. ");
        ru = 1;
        rtn = 1;
    }
    else{
        memcpy(&c_print, &h_c[0], sizeof(FP32));
        printText("Not fall in any cases, need to rethink");
        std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
    }
    
    /*
    Add negative numbers and rounding
    */
    host_reset(h_a, h_b, h_c);
    h_a[0] = minnormal16;
    h_a[1] = one16;
    h_b[0] = __float2half(-ldexp(1., -9) - ldexp(1., -10));
    h_b[1] = __float2half(-2.);
    wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == -2.){
        printText("inner product addition rounding up or rounding to zero. ");
        ru = ru & 1;
        rtz = rtz & 1;
        rtn = 0;
        rd = 0;
    }
    else if(h_c[0] == -2. - ldexp(1., -22)){
        printText("inner product addition rounding down or rounding to nearest. ");
        rtn = rtn & 1;
        rd = rd &1;
        ru = 0;
        rtz = 0;
    }
    else{
        printText("Not fall in any cases, need to rethink");
        memcpy(&c_print, &h_c[0], sizeof(FP32));
        std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
    }

    /*
     ========================== ==========================  Test Rounding mode for accumulation  ========================== ========================== 
    */
    bool artz=0, aru=0, ard=0, artn=0;
    printSubTest("Test initial rounding mode for accumulation.(a11b11+c11)");

    /*
    Add postive numbers and rounding
    */

    host_reset(h_a, h_b, h_c);
    h_a[0] = minnormal16;
    h_b[0] = __float2half(ldexp(1., -9) + ldexp(1., -10));
    h_c[0] =2.;
    wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == 2.){
        printText("accumulation rounding down or rounding to zero. ");
        artz = 1;
        ard = 1;
    }
    else if(h_c[0] == 2. + ldexp(1., -22)){
        printText("accumulation rounding up or rounding to nearest. ");
        artn = 1;
        aru = 1;
    }
    else{
        printText("Not fall in any cases, need to rethink");
        memcpy(&c_print, &h_c[0], sizeof(FP32));
        std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
    }
    
    /*
    Add negative numbers and rounding
    */
    host_reset(h_a, h_b, h_c);
    h_a[0] = one16;
    h_b[0] = __float2half(-2.);
    h_c[0] = -ldexp(1., -23) - ldexp(1., -24);
    wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == -2.){
        printText("accumulation rounding up or rounding to zero. ");
        aru = aru & 1;
        artz = artz & 1;
        ard = 0;
        artn = 0;
    }
    else if(h_c[0] == -2. - ldexp(1., -22)){
        printText("accumulation rounding down or rounding to nearest. ");
        artn = artn & 1;
        ard = ard & 1;
        aru = 0;
        artz = 0;
    }
    else{
        printText("Not fall in any cases, need to rethink");
        memcpy(&c_print, &h_c[0], sizeof(FP32));
        std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
    }
    
  
    /*
     ==========================  ========================== Test Extra bits ==========================  ==========================
    */

    printSubTest("Test extra bits");
    bool one_extr = 0, three_extr = 0; 
    /*
    Test 1 extra bits
    */
    host_reset(h_a, h_b, h_c);
    h_a[0] = one16;
    h_b[0] = one16;
    h_c[0] = -belowone;
    wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if(h_c[0] == ldexp(1.,-24)){
        printText("Use one extra bit (guard bit), and this can ensure the relative error is less than 2ulp");
        one_extr = 1;
    }
    else{
        printText("Not use any extra bits.");
    }

    /*
    Test 3 extra bits under tensor/matrix cores' rounding mode
    */

    if(rtz & artz & one_extr){
        host_reset(h_a, h_b, h_c);
        h_a[0] = one16;
        h_b[0] = __float2half(1. + ldexp(1., -2));
        h_c[0] = -(ldexp(1., -3) + ldexp(1., -25));
        wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        memcpy(&c_compare, h_c, sizeof(FP32));
        if(c_compare == 0x3F8FFFFF){
            printText("There are 2 or 3 extra bits");
        }
        else if(c_compare == 0x3F900000){
            printText("Only one extra bit");
            artn = 1;
        }
        else{
            printText("Not fall in any cases, need to rethink");
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
        }
    }
    /*
    Test 3 extra bits under rounding to nearest
    */
    else if(one_extr & rtn & artn) {
        host_reset(h_a, h_b, h_c);
        h_a[0] = one16;
        h_b[0] = __float2half(1. + ldexp(1., -2));
        h_c[0] = -(ldexp(1., -3) +ldexp(1., -24)+ldexp(1., -26));
        wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        memcpy(&c_compare, h_c, sizeof(FP32));
        if(c_compare == 0x3f8fffff){
            printText("There are 3 extra bits");
            three_extr = 1;
        }
        else if(c_compare == 0x3F900000){
            printText("One or two extra bits");
        }
        else{
            printText("Not fall in any cases, need to rethink");
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
        }
    }
    else{
        printf("Warning!!! Cannot draw conclusion from the current tests.");
    }

    /*
    If rounding to nearst and use extra 3 bits, if it is tie to even
    */
    printSubTest("Test if tie to even (only show results if three extra bits and rounding to nearest mode)");
    bool rtn_tie_to_even=0;
    if(three_extr & rtn & artn){
        host_reset(h_a, h_b, h_c);
        h_a[0] = ldexp(1, -10);
        h_b[0] = minnormal16;
        h_c[0] = 1.;
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        if(h_c[0] == 1.){
            printText("rtn is rounding to even when tie.");
            rtn_tie_to_even = 1;
        }
        else{
            printText("Not tie to even");
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
        }   
    }

     /*
    If the third bit is a stick bit (It will be set as one if there is a 1 after the third extra bit)
    */
    printSubTest("Test if third bit is a sticky bit (only show results if three extra bits)");
    bool stick_bit=0;
    if(three_extr & rtn & artn &rtn_tie_to_even ){
        host_reset(h_a, h_b, h_c);
        h_a[0] = one16;
        h_b[0] = one16;
        h_c[0] = ldexp(1.,-24) + ldexp(1.,-27);
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        memcpy(&c_compare, &h_c[0], sizeof(FP32));
        if(h_c[0] == 1+ldexp(1., -23)){
            printText("The third bit is a sticky bit and will memory the precision after it. ");
            stick_bit = 1;
        }
        else{
            printText("The third bit is not a sticky bit");
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
        }   
    }
    /*
    ==========================  ========================== Test FMA features for a11b11+a12b21+c11 ==========================  ========================== 
    */
    printMainTest("Test FMA features for a11b11+a12b21+c11");
    printSubTest("Test if extra bits are preserved and round at the end");
    // printheader(outfile, "Test accumulation order and if extra bits are preserved in the medium values");
    bool case1=0, case2 =0, case3 =0;
    bool preserve_intermed = 0;
    /*
    Test fell in case 1 or case 2 as the following printing demonstrate
    */
    host_reset(h_a, h_b, h_c);
    // for (index=0; index<2; index++) {
    //     h_a[index] = ldexp(1, -10);
    //     h_b[index] = minnormal16;
    // }

    FP32 sum_c;
    h_a[0] = ldexp(1, -10);
    h_b[0] = minnormal16;
    h_a[1] = ldexp(1, -10);
    h_b[1] = minnormal16;
    // h_a[2] = ldexp(1, -10);
    // h_b[2] = minnormal16;
    h_c[0] = 1.;
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    sum_c = h_c[0];

    for (index=0; index<2; index++) {
        h_c[0] = ldexp(1., -24);
        if (index>0){
            h_a[index-1] = ldexp(1, -10);
            h_b[index-1] = minnormal16;
        }
        h_a[index] = one16;
        h_b[index] = one16;
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        if(!(h_c[0] == sum_c)){
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
            memcpy(&c_print, &sum_c, sizeof(FP32));
            std::cout << "sum_c = " << std::hex << c_print << std::endl;
            break;
        }
    }
     if(index == 2 && h_c[0] == sum_c && sum_c == 1.){
        printText("Change the order of four elements won't change the value, but the intermediate result is not preserved.");
        case3 =1;
    }
    else if(index == 2 && h_c[0] == sum_c){
        printText("Change the order of four elements won't change the value, and the final result counting the ldexp(1., -24) . This will fall one of the two cases");
        printText( "(1) Extra bits are not preserved in the intermediate values but accumulate the smallest numbers together firstly.");
        printText("(2) Extra bits are preserved in the intermediate values and rounding at the end.");
        case1 = 1;
    }
    else{
        std::cout << "index is " << index << std::endl;
        printText("Change the order of four elements will change the value. This will fall in this case: ");
        // printitem(outfile, "(1) The minimum unit to preserve bits is less than 3.");
        printText("The accumulation order is under the user's control and the extra bits are not preserved in the partial sum");
        case2 = 1;
    }
    /*
    Test tensor cores rounding and use one extra bit
     in case1: accumulate from smallest numbers firstly or Extra bits are preserved in the intermediate values. 
    */
    if(case1 && rtz && artz && one_extr && !stick_bit){
        host_reset(h_a, h_b, h_c);
        h_a[0] = ldexp(1, -10);
        h_b[0] = minnormal16;
        h_a[1] = ldexp(1, -11);
        h_b[1] = minnormal16;
        h_a[2] = ldexp(1, -11);
        h_b[2] = minnormal16;
        h_c[0] = 1.;
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        if(h_c[0] == 1. ){
            printText("Not accumulate from the smallest number together firstly, meaning Extra bits are preserved in the intermediate values");
            preserve_intermed = 1; 
        }
        else if(h_c[0] == 1. + ldexp(1., -23)){
            printText("accumulate the smallest number together.");
        }
        else{
            printText("Not fall in any cases, need to rethink.");
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;            
        }
    }
    /*
    Test rounding to nearst and use three extra bit
    in case1: accumulate from smallest numbers firstly or Extra bits are preserved in the intermediate values. 
    */
    else if(case1 && rtn && artn && three_extr && rtn_tie_to_even && stick_bit){
        preserve_intermed = 1;
        printText("Since the device implement rounding to nearest and stick bit, it doesn't matter if accumulate the smallest numbers together or the extra bits are preserved. The precision is always preserved. ");
    }
    else if(case3 || case2){
        printText(" May normalize and rounding the partial sum. ");
    }
    else{
        printText("Warning!!! Cannot draw conclusion from the current tests.");
    }
    /*
    ==========================  ========================== Test the minumum unit to preserve the precision (extra bits) ==========================  ========================== 
    */
    int min_preserve_uint = 1; 

    printSubTest("Test FMA unit width (the minumum unit to preserve precision).");
    host_reset(h_a, h_b, h_c);
    h_c[0] = 1.;
    h_a[0] = ldexp(1, -10);
    h_b[0] = minnormal16;
    if(case1){
        for(index = 1; index < 16; index++){
            if(index > 1){
                h_a[index-1] = 0;
                h_b[index-1] = 0;
                h_c[0] = 1.;
            }
            h_a[index] = ldexp(1, -10);
            h_b[index] = minnormal16;
            wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
            if(h_c[0] == 1.){
                break;
            }
            else if(h_c[0]!=1+ldexp(1., -23)){
                printText("Not fall in any cases, need to rethink.");               
                memcpy(&c_print, &h_c[0], sizeof(FP32));
                std::cout << "h_c[0] = " << std::hex << c_print << std::endl;
            }
        }
        if(index < 16){
            std::string out= "The FMA unit width is " + std::to_string(index);
            printText(out.c_str());
            min_preserve_uint = index;
        }
        else{
            min_preserve_uint = 16;
            printText("The FMA unit width is larger than 16.");
        }
    }
    else if(case3 || case2){
        min_preserve_uint = 1;
        printText("The FMA unit width is 1, meaning no FMA feature.");
    }
    /*
    ==========================  ========================== Test if normalize once at the end or normalize the partial sum  ==========================  ========================== 
    */
    printSubTest("Test if normalize once at the end or normalize the partial sum.");
    bool norm_once = 0, norm_interm = 0;
    if((min_preserve_uint >=4 ) & preserve_intermed & rtz&artz & one_extr &!three_extr){
        host_reset(h_a, h_b, h_c);
        h_a[0] = ldexp(1, -9);
        h_b[0] = minnormal16;
        h_a[1] = ldexp(1, -11);
        h_b[1] = minnormal16;
        h_a[2] = ldexp(1, -11);
        h_b[2] = minnormal16;
        h_c[0] = 1.-ldexp(1., -24);
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        if(h_c[0] == 1. ){
            printText("Normalize the partial sum.");
            norm_interm = 1;
        }
        else if(h_c[0] == 1. + ldexp(1., -23)){
            printText("Normalize once at the end. ");
            norm_once =1;
        }
        else{
            printText("Not fall in any cases, need to rethink.");               
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;            
        }
        
    }
    else if((min_preserve_uint >=4) & preserve_intermed & rtn &artn &rtn_tie_to_even &stick_bit & three_extr){
        printText("Since the device implement rounding to nearst and stick bit, Both method will preserve the full precision");
    }
    else if(!preserve_intermed & rtn &artn &rtn_tie_to_even){
        host_reset(h_a, h_b, h_c);
        h_a[0] = ldexp(1, -9);
        h_b[0] = minnormal16;
        h_a[1] = ldexp(1, -10);
        h_b[1] = minnormal16;
        // h_a[2] = ldexp(1, -11);
        // h_b[2] = minnormal16;
        h_c[0] = 1.-ldexp(1., -24);
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        if(h_c[0] == 1. ){
            printText("Normalize the partial sum.");
            norm_interm = 1;
        }
        else if(h_c[0] == 1. + ldexp(1., -23)){
            printText("Normalize once at the end. ");
            norm_once =1;
        }
        else{
            printText("Not fall in any cases, need to rethink.");               
            memcpy(&c_print, &h_c[0], sizeof(FP32));
            std::cout << "h_c[0] = " << std::hex << c_print << std::endl;            
        }
    }
    else{
        printf("Warning!!! Cannot draw conclusion from the current tests.");
    }
     /*
    ==========================  ========================== Test Monotonicity of addition  ==========================  ========================== 
    */
    printMainTest("Test if the monotonicity is preserved for multi-term addition");
    bool mon = 0;

    if(norm_once & (min_preserve_uint >= 8) & one_extr &!three_extr){
        host_reset(h_a, h_b, h_c);
        h_a[0] = one16;
        h_b[0] = one16;
        h_a[1] = one16;
        h_b[1] = one16;
        h_a[2] = one16;
        h_b[2] = one16;
        h_a[3] = one16;
        h_b[3] = one16;
        h_a[4] = one16;
        h_b[4] = one16;
        h_a[5] = one16;
        h_b[5] = one16;
        h_a[6] = one16;
        h_b[6] = one16;
        h_a[7] = one16;
        h_b[7] = one16;
        h_c[0] = ldexp(1,25) - ldexp(1,1);
                   
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        float smaller_c = h_c[0];
       
        h_c[0] = ldexp(1,25);
        wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        float larger_c = h_c[0];
        memcpy(&c_print, &larger_c, sizeof(FP32));
        // std::cout << "larger_c = " << std::hex << c_print << std::endl; 
        if(smaller_c <=  larger_c){
            printText("Monotonicity is preserved.");
            mon = 1;
        }
        else{
            printText("Monotonicity is not preserved.");
        }
    
   }
   else if((min_preserve_uint >= 2) & three_extr & stick_bit){
        printText("Since the device implement rounding to nearest and stick bit, monotonicity is always preserved");
   }
   else if(norm_interm & !preserve_intermed){
        printText("Since normalize the intermediate sum and the extra bits are not preserved, from mantis' proof, monotonicity is preserved. ");
   }
   else{
        printText("Warning!!! Cannot draw conclusion from the current tests.");
    }

    /*
     ==========================  ========================== Test Rounding mode for inner product addition (fp16 output) ==========================  ========================== 
    */

    bool fp16_rtz=0, fp16_ru=0, fp16_rd=0, fp16_rtn=0;
    printMainTest("Test the rounding mode when output is in fp16");

    /*
    Add postive numbers and rounding
    */
    printSubTest("Test the initial rounding mode for a11b11+a12b21");
    host_reset(h_a, h_b, h16_c);
    h_a[0] = one16;
    h_a[1] = one16;
    h_b[0] = 2.;
    h_b[1] = __float2half(ldexp(1, -10) + ldexp(1, -11));
    wmma_init_run (h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
    if(__half2float(h16_c[0]) == 2.){
        printText("inner product addition rounding down or rounding to zero.");
        fp16_rd = 1;
        fp16_rtz = 1;
    }
    else if(__half2float(h16_c[0]) == 2 + ldexp(1,-9)){
        printText("inner product addition rounding up or rounding to nearest. ");
        fp16_rtn = 1;
        fp16_ru =1;
    }
    else{
        printText("Not fall in any cases, need to rethink.");               
        memcpy(&c_print, &h16_c[0], sizeof(FP16));
        std::cout << "h16_c[0] = " << std::hex << c_print << std::endl;
    }
    
    /*
    Add negative numbers and rounding
    */
    host_reset(h_a, h_b, h16_c);
    h_a[0] = one16;
    h_a[1] = one16;
    h_b[0] = -2.;
    h_b[1] = __float2half(-ldexp(1, -10) - ldexp(1, -11));
    wmma_init_run (h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
    if(__half2float(h16_c[0]) == -2.){
        printText("inner product addition rounding up or rounding to zero. ");
        fp16_rd = 0;
        fp16_rtn = 0;
        fp16_ru = fp16_ru &1;
        fp16_rtz = fp16_rtz &1;
    }
    else if(__half2float(h16_c[0]) ==  - 2 - ldexp(1, -9)){
        printText("inner product addition rounding down or rounding to nearest. ");
        fp16_ru = 0;
        fp16_rtz = 0;
        fp16_rd = fp16_rd &1;
        fp16_rtn = fp16_rtn &1;
    }
    else{
        printText("Not fall in any cases, need to rethink.");               
        memcpy(&c_print, &h16_c[0], sizeof(FP16));
        std::cout << "h16_c[0] = " << std::hex << c_print << std::endl;
    }
    /*
     ========================== ==========================  Test Rounding mode for accumulation  ========================== ========================== 
    */
    bool fp16_rtza=0, fp16_rua=0, fp16_rda=0, fp16_rtna=0;
    printSubTest("Test the initial rounding mode for a11b11+c11 when output is fp16");

    /*
    Add postive numbers and rounding
    */
    host_reset(h_a, h_b, h16_c);
    h_a[0] = one16;
    h_b[0] = __float2half(ldexp(1., -10) + ldexp(1., -11));
    h16_c[0] =__float2half(2.);
    wmma_init_run (h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
    if(__half2float(h16_c[0]) == 2.){
        printText("accumulation rounding down or rounding to zero. ");
        fp16_rda = 1;
        fp16_rtza = 1;
    }
    else if(__half2float(h16_c[0]) == 2 + ldexp(1,-9)){
        printText("accumulation rounding up or rounding to nearest. ");
        fp16_rtna = 1;
        fp16_rua =1;
    }
    else{
        printText("Not fall in any cases, need to rethink.");                       
        memcpy(&c_print, &h16_c[0], sizeof(FP16));
        std::cout << "h16_c[0] = " << std::hex << c_print << std::endl;
    }
    
    /*
    Add negative numbers and rounding
    */
    host_reset(h_a, h_b, h16_c);
    h_a[0] = one16;
    h_b[0] = __float2half(-2.);
    h16_c[0] = __float2half(-ldexp(1., -10) - ldexp(1., -11));
    wmma_init_run (h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
    if(__half2float(h16_c[0]) == -2.){
        printText("accumulation rounding up or rounding to zero. ");
        fp16_rda = 0;
        fp16_rtna = 0;
        fp16_rua = fp16_rua &1;
        fp16_rtza = fp16_rtza &1;
    }
    else if(__half2float(h16_c[0]) ==  - 2 - ldexp(1, -9)){
        printText("accumulation rounding down or rounding to nearest. ");
        fp16_rua = 0;
        fp16_rtza = 0;
        fp16_rda = fp16_rda &1;
        fp16_rtna = fp16_rtna &1;
    }
    else{
        printText("Not fall in any cases, need to rethink.");               
        memcpy(&c_print, &h16_c[0], sizeof(FP16));
        std::cout << "h16_c[0] = " << std::hex << c_print << std::endl;
    }
    /*
     ==========================  ========================== Test if fp16 output, then the full computation is in full 32 precision (at least three extra bits) ==========================  ========================== 
    */
     bool compute_in_32 = 0;
     printMainTest("Test if fp16 output, then the full computation is in full 32 precision (at least three extra bits)");
    
    if(fp16_rtz & fp16_rtza){

        host_reset(h_a, h_b, h16_c);
        h_a[0] = one16;
        h_a[1] = __float2half(-ldexp(1, -9));
        h_b[0] = one16;
        h_b[1] = minnormal16;
        wmma_init_run (h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
        if(__half2float(h16_c[0]) == 1.){
            printText("if output fp16, the full computation is not performed in full32 precision and not at least three extra bits. ");
        }
        else if(__half2float(h16_c[0]) == 1.- ldexp(1, -10)){
            printText("if output fp16, the full computation is performed in full32 precision, so at least three extra bits. ");
            compute_in_32 = 1;
        }
        else{
            printText("Not fall in any cases, need to rethink.");               
            memcpy(&c_print, &h16_c[0], sizeof(FP16));
            std::cout << "h16_c[0] = " << std::hex << c_print << std::endl;
        }
    }

    else if(fp16_rtn & fp16_rtna){
        host_reset(h_a, h_b, h16_c);
        h_a[0] = one16;
        h_a[1] = one16;
        h_b[0] = one16;
        h_b[1] = __float2half(ldexp(1, -11));
        wmma_init_run (h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
        FP16 c1 = h16_c[0];
        h_b[1] = __float2half(ldexp(1, -11) + ldexp(1, -14));
        h16_c[0] = minnormal16;
        wmma_init_run (h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
        FP16 c2 = h16_c[0];
        if(__half2float(c1) != __half2float(c2)){
            printText("if output fp16, the full computation is performed in full32 precision, so at least three extra bits and tie to even.");
            compute_in_32 = 1;
        }
        else if(__half2float(c1) == 1.+ldexp(1, -10)){
            printText("if output fp16, the full computation is not performed in full32 precision and not at least three extra bits, or not tie to even. ");
        }
        else{
            printText("Not fall in any cases, need to rethink.");               
            memcpy(&c_print, &h16_c[0], sizeof(FP16));
            std::cout << "h16_c[0] = " << std::hex << c_print << std::endl;
        }

    }
    else{
        printf("Warning!!! Cannot draw conclusion from the current tests.");
    }
    summerize(GPUModel, support_sub_input,
                rtz, ru, rd, rtn,
                artz, aru, ard, artn,
                one_extr, three_extr, rtn_tie_to_even, stick_bit,
                min_preserve_uint,
                fp16_rtz, fp16_ru, fp16_rd, fp16_rtn);


}
